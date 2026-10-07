from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Mengambil absolute path agar Vercel bisa menemukan file model
base_dir = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(base_dir, 'model_ad_click.pkl'), 'rb'))
kolom_fitur = pickle.load(open(os.path.join(base_dir, 'kolom_fitur.pkl'), 'rb'))

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Buat DataFrame awal dengan nilai 0
        input_data = pd.DataFrame(0, index=[0], columns=kolom_fitur)
        
        # Masukkan data dari request (Frontend)
        input_data.at[0, 'Daily Time Spent on Site'] = float(data['time_on_site'])
        input_data.at[0, 'Age'] = int(data['age'])
        input_data.at[0, 'Area Income'] = float(data['income'])
        input_data.at[0, 'Daily Internet Usage'] = float(data['internet_usage'])
        input_data.at[0, 'Male'] = int(data['gender'])
        input_data.at[0, 'Hour'] = int(data['jam_tayang'])
        
        # Mapping hari tayang
        hari_dict = {"Senin":0, "Selasa":1, "Rabu":2, "Kamis":3, "Jumat":4, "Sabtu":5, "Minggu":6}
        day_of_week = hari_dict[data['hari']]
        input_data.at[0, 'DayOfWeek'] = day_of_week
        input_data.at[0, 'Is_Weekend'] = 1 if day_of_week >= 5 else 0
        
        # Handle Kategori
        kategori_col = f"category_{data['kategori']}"
        if kategori_col in kolom_fitur:
            input_data.at[0, kategori_col] = 1
            
        # Prediksi
        probabilitas = model.predict_proba(input_data)[0][1]
        prediksi = int(model.predict(input_data)[0])
        
        return jsonify({
            "status": "success",
            "probabilitas": round(probabilitas * 100, 2),
            "prediksi": prediksi
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Dibutuhkan oleh Vercel
if __name__ == "__main__":
    app.run(debug=True)
