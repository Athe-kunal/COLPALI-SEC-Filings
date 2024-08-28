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

Then you can ask questions and get token level attention maps

![Token-level Attention Maps](https://github.com/Athe-kunal/COLPALI-SEC-Filings/blob/main/assets/Screen%20Shot%202024-08-28%20at%2012.48.31%20AM.png)