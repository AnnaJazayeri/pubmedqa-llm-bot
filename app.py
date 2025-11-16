import os
import requests
import streamlit as st
import numpy as np
import pandas as pd
import torch

# Google Drive download links
URL_CONTEXT = "https://drive.google.com/uc?export=download&id=1e4BwDZaqNPe-8i3KHZskvIlVIUdRb6Fb" # https://drive.google.com/file/d/1e4BwDZaqNPe-8i3KHZskvIlVIUdRb6Fb/view?usp=sharing
URL_PARQUET = "https://drive.google.com/uc?export=download&id=1zx4RAR_csv4sutBDg2RuNcWH-UxR1WoB" # https://drive.google.com/file/d/1zx4RAR_csv4sutBDg2RuNcWH-UxR1WoB/view?usp=sharing
URL_MODEL = "https://drive.google.com/uc?export=download&id=1RNPiTK52eXe47WyKANLxMZ1OiAtS4eo5" # https://drive.google.com/file/d/1RNPiTK52eXe47WyKANLxMZ1OiAtS4eo5/view?usp=sharing

def download_file(url, filename):
    if not os.path.exists(filename):
        with st.spinner(f"Downloading {filename} ..."):
            r = requests.get(url, allow_redirects=True)
            open(filename, 'wb').write(r.content)
            st.success(f"{filename} downloaded.")

# Ensure files exist
download_file(URL_CONTEXT, "context_embs_pubmedqa.npy")
download_file(URL_PARQUET, "df_all_pubmedqa.parquet")
download_file(URL_MODEL, "dual_encoder_pubmedqa.pt")

