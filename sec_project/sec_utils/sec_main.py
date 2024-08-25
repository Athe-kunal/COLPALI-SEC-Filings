from sec_filings_to_pdf import sec_save_pdfs

filing_types = ["10-K","10-Q"]
include_amends = True
html_urls, metadata_json, metadata_file_path,input_ticker_year_path = sec_save_pdfs(
        "GOOG", "2024", filing_types, include_amends
    )
