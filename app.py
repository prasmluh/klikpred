import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load model dan kolom
model = pickle.load(open('model_ad_click.pkl', 'rb'))
kolom_fitur = pickle.load(open('kolom_fitur.pkl', 'rb'))

st.title("🎯 Ad-Targeting CTR Prediction")
st.write("Prediksi apakah user akan mengklik iklan (Support by Hugging Face)")

# Input dari User
age = st.slider("Usia", 18, 70, 30)
income = st.number_input("Area Income", value=400000000)
time_on_site = st.number_input("Waktu di Web (menit)", value=60.0)
internet_usage = st.number_input("Internet Usage (menit)", value=200.0)
gender = st.selectbox("Gender", ["Perempuan", "Laki-Laki"])
kategori = st.selectbox("Kategori Iklan", ["Fashion", "Food", "Electronic", "Travel", "House", "Furniture", "Otomotif", "Finance", "Bank", "Health"])

if st.button("Prediksi Klik"):
    # Preprocessing sederhana
    gender_val = 0 if gender == "Perempuan" else 1
    input_data = pd.DataFrame(0, index=[0], columns=kolom_fitur)
    
    input_data['Daily Time Spent on Site'] = time_on_site
    input_data['Age'] = age
    input_data['Area Income'] = income
    input_data['Daily Internet Usage'] = internet_usage
    input_data['Male'] = gender_val
    
    cat_col = f'category_{kategori}'
    if cat_col in input_data.columns:
        input_data[cat_col] = 1
        
    # Prediksi
    prob = model.predict_proba(input_data)[0][1]
    
    if prob > 0.5:
        st.success(f"User berpotensi KLIK IKLAN! (Probabilitas: {prob*100:.2f}%)")
    else:
        st.warning(f"User tidak akan klik iklan. (Probabilitas: {prob*100:.2f}%)")
