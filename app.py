import streamlit as st
import pandas as pd
import pickle
import os

# Mendapatkan lokasi folder tempat app.py berada
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Menggabungkan path dengan nama file
MODEL_PATH = os.path.join(BASE_DIR, 'model_ad_click.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'kolom_fitur.pkl')

# Load model dan kolom dengan path yang sudah pasti
model = pickle.load(open(MODEL_PATH, 'rb'))
kolom_fitur = pickle.load(open(FEATURES_PATH, 'rb'))
