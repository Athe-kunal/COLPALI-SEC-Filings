from functools import partial
import streamlit as st
from sec_project.backend import index_images
from sec_project.colpali_utils import search
from colpali_engine.interpretability.processor import ColPaliProcessor


@st.cache_resource
def build_index(ticker,year,filing_types):
    ds, images, metadata, model, device, processor = index_images(ticker=ticker,year=year,filing_types=filing_types,include_amends=True)
    search_index = partial(search,ds,processor,device,model,3)
    colpali_processor = ColPaliProcessor(processor=processor)
    return search_index, images, metadata, model, colpali_processor 


ticker = st.text_input(label="Ticker")
year = st.text_input(label="Year")


if ticker != "" and year != "":
    search_index, images, metadata, model, colpali_processor  = build_index(ticker, year, ["10-K","10-Q"])

    st.session_state['search_index'] = search_index
    st.session_state['images'] = images
    st.session_state['metadata'] = metadata
    st.session_state['model'] = model
    st.session_state['colpali_processor'] = colpali_processor
    st.session_state['ticker'] = ticker
    st.session_state['year'] = year
