import streamlit as st
from sec_project.colpali_utils import get_attention_maps
from colpali_engine.interpretability.plot_utils import plot_attention_heatmap


search_index = st.session_state['search_index']
images = st.session_state['images']
metadata = st.session_state['metadata']
ticker = st.session_state['ticker']
year = st.session_state['year']
model = st.session_state['model']
colpali_processor = st.session_state['colpali_processor']

st.title(f"{ticker}-{year}")

query = st.text_input(label="Query")

def show_attention_maps():
    text_token_dict = st.session_state['text_token_dict']
    text_token_id = text_token_dict[st.session_state['token']]
    idx_dict = st.session_state['idx_dict']
    figs = []
    for image_idx,attention_maps in idx_dict.items():
        fig,ax = plot_attention_heatmap(
            images[int(image_idx)],
            patch_size = 14,
            image_resolution=448,
            attention_map=attention_maps[0].float().squeeze(axis=0)[text_token_id]
            )
        figs.append(fig)
    return figs

if query != "":
    relevant_idxs = search_index(query)

    idx_dict = {str(idx):None for idx in relevant_idxs}
    st.session_state['idx_dict'] = idx_dict
    for idx in relevant_idxs:
        attention_maps_normalized, attention_maps, text_tokens = get_attention_maps(
            model=model,
            processor=colpali_processor,
            query=query,
            image=images[idx],
        )
        idx_dict[str(idx)] = [attention_maps_normalized,attention_maps]
    
    text_token_dict = {text_tokens[ix]:ix for ix in range(len(text_tokens))}
    st.session_state['text_token_dict'] = text_token_dict
    text_token = st.selectbox(
        "TEXT TOKENS",
        tuple(text_tokens),
        on_change=show_attention_maps
    )
    
    st.session_state['token'] = text_token
    
    figs = show_attention_maps()
    

    for fig in figs:
        st.pyplot(fig)