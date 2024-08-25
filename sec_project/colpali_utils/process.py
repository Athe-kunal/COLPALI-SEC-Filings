from PIL import Image, ImageChops
import numpy as np
import torch
import os
from pdf2image import convert_from_path
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from colpali_engine.trainer.retrieval_evaluator import CustomEvaluator
from colpali_engine.utils.colpali_processing_utils import process_images, process_queries

mock_image = Image.new("RGB", (448, 448), (255, 255, 255))

def search(ds,processor, device,model,k:int,query:str):
    qs = []
    with torch.no_grad():
        batch_query = process_queries(processor,[query],mock_image)
        batch_query = {k: v.to(device) for k,v in batch_query.items()}
        embeddings_query = model(**batch_query)
        qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))
    
    retriever_evaluator = CustomEvaluator(is_multi_vector=True)
    scores = retriever_evaluator.evaluate(qs,ds)
    best_pages_idxs = np.argsort(scores,axis=1).squeeze(0)
    return best_pages_idxs[::-1][:k]


def trim_and_square(im, padding=10, target_size=448):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        # Add padding to the bounding box
        left, top, right, bottom = bbox
        left = max(0, left - padding)
        top = max(0, top - padding)
        right = min(im.width, right + padding)
        bottom = min(im.height, bottom + padding)
        trimmed = im.crop((left, top, right, bottom))
    else:
        trimmed = im

    # Resize to square
    w, h = trimmed.size
    if w > h:
        new_w = target_size
        new_h = int(h * (target_size / w))
    else:
        new_h = target_size
        new_w = int(w * (target_size / h))
    
    resized = trimmed.resize((new_w, new_h), Image.LANCZOS)

    # Create a white square image
    square = Image.new('RGB', (target_size, target_size), (255, 255, 255))
    
    # Paste the resized image onto the square, centered
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    square.paste(resized, (paste_x, paste_y))

    return square

def index(pdf_paths,ds,processor,model,device,batch_size:int=32):
    images = []
    metadata = []
    for f in pdf_paths:
        file_name = f.rsplit("/",1)[-1]
        f_imgs = convert_from_path(f)
        for idx,f_img in enumerate(f_imgs):
            cropped_f_img = trim_and_square(f_img)
            images.append(cropped_f_img)
            metadata.append({"file":file_name,"page_num":idx})
            
    dataloader = DataLoader(
        images,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: process_images(processor,x)
    )
    
    for batch_doc in tqdm(dataloader):
        with torch.no_grad():
            batch_doc = {k: v.to(device) for k,v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)
        ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
    print(f"Uploaded and converted {len(images)} pages")
    return ds, images, metadata

def read_pdfs(dir:str):
    files = []
    for file in os.listdir(dir):
        if file.endswith(".pdf"):
            files.append(os.path.join(dir,file))
    return files