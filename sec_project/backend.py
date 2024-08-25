import os
from typing import List
from sec_project.sec_utils import sec_save_pdfs
from sec_project.colpali_utils import search, index, read_pdfs, load_model

def index_images(ticker:str,year:str,filing_types:List[str],include_amends:bool):
    ticker = ticker.upper()
    html_urls, metadata_json, metadata_file_path,input_ticker_year_path = sec_save_pdfs(
        ticker, year, filing_types, include_amends
    )
    print("Saved the pdfs")
    
    pdf_dir = os.path.join("output","SEC_EDGAR_FILINGS",f"{ticker}-{year}")
    pdf_paths = read_pdfs(pdf_dir)
    
    dataset = []
    model, device, processor = load_model()
    ds, images, metadata = index(pdf_paths,dataset,processor,model,batch_size=32,device=device)
    return ds, images, metadata, model, device, processor

