import pprint
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import matplotlib.pyplot as plt
import torch
from einops import rearrange
from PIL import Image
from tqdm import trange

from colpali_engine.interpretability.plot_utils import plot_patches
from colpali_engine.interpretability.processor import ColPaliProcessor
from colpali_engine.interpretability.torch_utils import normalize_attention_map_per_query_token
from colpali_engine.interpretability.vit_configs import VIT_CONFIG
from colpali_engine.models.paligemma_colbert_architecture import ColPali

@dataclass
class InterpretabilityInput:
    query: str
    image: Image.Image
    start_idx_token: int
    end_idx_token: int

def get_attention_maps(
    model: ColPali,
    processor: ColPaliProcessor,
    query: str,
    image: Image.Image,
    add_special_prompt_to_doc: bool = True,
) -> None:

    # Sanity checks
    if len(model.active_adapters()) != 1:
        raise ValueError("The model must have exactly one active adapter.")

    if model.config.name_or_path not in VIT_CONFIG:
        raise ValueError("The model must be referred to in the VIT_CONFIG dictionary.")
    vit_config = VIT_CONFIG[model.config.name_or_path]
    # Preprocess the inputs
    input_text_processed = processor.process_text(query).to(model.device)
    input_image_processed = processor.process_image(image, add_special_prompt=add_special_prompt_to_doc).to(
        model.device
    )

    # Forward pass
    with torch.no_grad():
        output_text = model.forward(**asdict(input_text_processed))  # (1, n_text_tokens, hidden_dim)

    # NOTE: `output_image`` will have shape:
    # (1, n_patch_x * n_patch_y, hidden_dim) if `add_special_prompt_to_doc` is False
    # (1, n_patch_x * n_patch_y + n_special_tokens, hidden_dim) if `add_special_prompt_to_doc` is True
    with torch.no_grad():
        output_image = model.forward(**asdict(input_image_processed))

    if add_special_prompt_to_doc:  # remove the special tokens
        output_image = output_image[
            :, : processor.processor.image_seq_length, :
        ]  # (1, n_patch_x * n_patch_y, hidden_dim)

    output_image = rearrange(
        output_image, "b (h w) c -> b h w c", h=vit_config.n_patch_per_dim, w=vit_config.n_patch_per_dim
    )  # (1, n_patch_x, n_patch_y, hidden_dim)

    # Get the unnormalized attention map
    attention_map = torch.einsum(
        "bnk,bijk->bnij", output_text, output_image
    )  # (1, n_text_tokens, n_patch_x, n_patch_y)
    attention_map_normalized = normalize_attention_map_per_query_token(
        attention_map
    )  # (1, n_text_tokens, n_patch_x, n_patch_y)
    attention_map_normalized = attention_map_normalized.float()

    # Get text token information
    text_tokens = processor.tokenizer.tokenize(processor.decode(input_text_processed.input_ids[0]))
    # print("Text tokens:")
    # pprint.pprint(text_tokens)
    # print("\n")

    return attention_map_normalized, attention_map,text_tokens