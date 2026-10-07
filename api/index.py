from flask import Flask, request, jsonify
import numpy as np
import xgboost as xgb
import json
import os

app = Flask(__name__)

# Konfigurasi path file JSON
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model_ad_click.json')
FEATURES_PATH = os.path.join(BASE_DIR, 'kolom_fitur.json')

# Load Kolom Fitur
try:
    with open(FEATURES_PATH, 'r') as f:
        kolom_fitur = json.load(f)
except Exception as e:
    kolom_fitur = None
    print("Gagal memuat kolom fitur:", e)

# Load Model XGBoost langsung tanpa Scikit-Learn
try:
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
except Exception as e:
    model = None
    print("Gagal memuat model:", e)

@app.route('/api/predict', methods=['POST'])
def predict():
    if not model or not kolom_fitur:
        return jsonify({"status": "error", "message": "Model tidak tersedia"}), 500
        
    try:
        data = request.json
        
        # 1. Buat Array kosong berisi angka 0, sepanjang jumlah kolom fitur
        input_array = np.zeros((1, len(kolom_fitur)))
        
        # Fungsi bantuan untuk mengisi nilai array berdasarkan nama kolom
        def set_value(col_name, value):
            if col_name in kolom_fitur:
                idx = kolom_fitur.index(col_name)
                input_array[0, idx] = value

        # 2. Masukkan data input user ke dalam Array
        set_value('Daily Time Spent on Site', float(data['time_on_site']))
        set_value('Age', int(data['age']))
        set_value('Area Income', float(data['income']))
        set_value('Daily Internet Usage', float(data['internet_usage']))
        set_value('Male', int(data['gender']))
        set_value('Hour', int(data['jam_tayang']))
        
        # 3. Proses kolom Hari & Weekend
        hari_dict = {"Senin":0, "Selasa":1, "Rabu":2, "Kamis":3, "Jumat":4, "Sabtu":5, "Minggu":6}
        day_of_week = hari_dict.get(data['hari'], 0)
        
        set_value('DayOfWeek', day_of_week)
        set_value('Is_Weekend', 1 if day_of_week >= 5 else 0)
        
        # 4. Proses kolom Kategori Iklan
        kategori_col = f"category_{data['kategori']}"
        set_value(kategori_col, 1)
            
        # 5. Prediksi menggunakan XGBoost
        probabilitas = model.predict_proba(input_array)[0][1]
        prediksi = int(model.predict(input_array)[0])
        
        return jsonify({
            "status": "success",
            "probabilitas": round(float(probabilitas) * 100, 2),
            "prediksi": prediksi
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
