# COLPALI-SEC-Filings

This repo integrates [ColPali](https://github.com/illuin-tech/colpali/tree/main) to SEC filings. To setup start with installing the packages

```bash
poetry install
```

Now you can spin up the streamlit server

```bash
poetry run strealit run streamlit.py
```

It will ask for Ticker and Year. In the backend, it will download the data and then index it using SigLiP encoder.

Then you can as