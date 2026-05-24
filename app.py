from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# 1. Load PAKET MODEL BARU
try:
    export_package = joblib.load('model_triage_clinical.pkl')
    model = export_package['model']
    medians = export_package['medians']
except Exception as e:
    print(f"Gagal load model: {e}. Pastikan forest.py sudah dijalankan.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        base_keys = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain']
        
        patient_vitals = {}
        missing_fields = []

        # 2. LOGIKA IMPUTASI
        for key in base_keys:
            val = data.get(key)
            if val is None or val == "":
                missing_fields.append(key)
                patient_vitals[key] = medians[key]
            else:
                if key == 'temperature':
                    patient_vitals[key] = (float(val) * 9/5) + 32 # Suhu jadi Fahrenheit
                else:
                    patient_vitals[key] = float(val)

        # 3. HITUNG FITUR KLINIS UTAMA
        heartrate = patient_vitals['heartrate']
        sbp = patient_vitals['sbp']
        dbp = patient_vitals['dbp']
        shock_index = heartrate / sbp if sbp > 0 else 0
        MAP = (sbp + 2 * dbp) / 3
        
        feature_columns = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain', 'shock_index', 'MAP']
        features = pd.DataFrame([[
            patient_vitals['temperature'], heartrate, patient_vitals['resprate'], 
            patient_vitals['o2sat'], sbp, dbp, patient_vitals['pain'], 
            shock_index, MAP
        ]], columns=feature_columns)
        
        # 4. PREDIKSI UTAMA & CONFIDENCE
        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        confidence_percentage = round(max(probabilities) * 100, 1)

        # ====================================================================
        # 5. LOGIKA BARU: 3 SKENARIO SIMULASI (DENGAN JUDUL DINAMIS)
        # ====================================================================
        hypotheticals = []
        is_incomplete = len(missing_fields) > 0

        if is_incomplete:
            # Gunakan list agar lebih mudah dikelola (tanpa nama kaku di awal)
            scenarios_data = [
                {
                    'id': 'buruk',
                    'vals': {'temperature': 39.5, 'heartrate': 120.0, 'resprate': 28.0, 'o2sat': 92.0, 'sbp': 85.0, 'dbp': 50.0, 'pain': 8.0},
                    'desc': {'temperature': 'demam tinggi', 'heartrate': 'jantung berdebar sangat cepat', 'resprate': 'napas terengah-engah', 'o2sat': 'pernapasan terasa berat', 'sbp': 'pusing hebat (tensi anjlok)', 'dbp': 'keringat dingin', 'pain': 'rasa nyeri hebat tak tertahankan'}
                },
                {
                    'id': 'menengah',
                    'vals': {'temperature': 37.8, 'heartrate': 95.0, 'resprate': 20.0, 'o2sat': 96.0, 'sbp': 110.0, 'dbp': 70.0, 'pain': 4.0},
                    'desc': {'temperature': 'demam ringan', 'heartrate': 'jantung sedikit berdebar', 'resprate': 'napas agak cepat', 'o2sat': 'napas kurang lega', 'sbp': 'tensi agak rendah', 'dbp': 'badan lemas', 'pain': 'nyeri sedang'}
                },
                {
                    'id': 'sehat',
                    'vals': {'temperature': 36.5, 'heartrate': 75.0, 'resprate': 16.0, 'o2sat': 99.0, 'sbp': 120.0, 'dbp': 80.0, 'pain': 0.0},
                    'desc': {'temperature': 'suhu tubuh normal', 'heartrate': 'detak jantung normal', 'resprate': 'napas teratur', 'o2sat': 'pernapasan sangat lega', 'sbp': 'tensi normal', 'dbp': 'tidak ada keringat dingin/lemas', 'pain': 'tidak ada nyeri sama sekali'}
                }
            ]

            # Mapping nama status berdasarkan tingkat Acuity
            acuity_titles = {
                1: 'Kondisi Kritis',
                2: 'Risiko Tinggi',
                3: 'Kondisi Mendesak',
                4: 'Kondisi Stabil',
                5: 'Tidak Gawat'
            }

            for sc_data in scenarios_data:
                hypo_vitals = patient_vitals.copy()
                symptom_texts = []

                for field in missing_fields:
                    val = sc_data['vals'][field]
                    if field == 'temperature':
                        hypo_vitals[field] = (val * 9/5) + 32
                    else:
                        hypo_vitals[field] = val
                    symptom_texts.append(sc_data['desc'][field])

                h_heartrate = hypo_vitals['heartrate']
                h_sbp = hypo_vitals['sbp']
                h_dbp = hypo_vitals['dbp']
                h_shock_index = h_heartrate / h_sbp if h_sbp > 0 else 0
                h_MAP = (h_sbp + 2 * h_dbp) / 3

                h_features = pd.DataFrame([[
                    hypo_vitals['temperature'], h_heartrate, hypo_vitals['resprate'],
                    hypo_vitals['o2sat'], h_sbp, h_dbp, hypo_vitals['pain'],
                    h_shock_index, h_MAP
                ]], columns=feature_columns)

                h_pred = int(model.predict(h_features)[0])
                h_prob = round(max(model.predict_proba(h_features)[0]) * 100, 1)

                is_abnormal = (
                    hypo_vitals['temperature'] > 100.04 or hypo_vitals['temperature'] < 95.9 or 
                    hypo_vitals['heartrate'] > 100 or hypo_vitals['heartrate'] < 50 or
                    hypo_vitals['resprate'] > 22 or hypo_vitals['resprate'] < 10 or
                    hypo_vitals['o2sat'] < 95 or
                    hypo_vitals['sbp'] > 140 or hypo_vitals['sbp'] < 90 or
                    hypo_vitals['pain'] > 3
                )

                if sc_data['id'] == 'sehat' and not is_abnormal and h_pred < 4:
                    h_pred = 4
                    h_prob = max(h_prob, 85.0)

                if len(symptom_texts) > 1:
                    symptoms_str = ", ".join(symptom_texts[:-1]) + ", dan " + symptom_texts[-1]
                else:
                    symptoms_str = symptom_texts[0]

                # Ambil judul yang pas dengan angka Acuity-nya
                dynamic_title = acuity_titles.get(h_pred, 'Status Kondisi')

                hypotheticals.append({
                    'title': dynamic_title,
                    'symptoms': symptoms_str,
                    'acuity': h_pred,
                    'confidence': h_prob
                })

            # MENGURUTKAN SIMULASI BERDASARKAN AKURASI TERTINGGI (Descending)
            hypotheticals.sort(key=lambda x: x['confidence'], reverse=True)

        return jsonify({
            'acuity': prediction,
            'confidence': confidence_percentage,
            'is_incomplete': is_incomplete,
            'hypotheticals': hypotheticals
        })

    except Exception as e:
        print(f"❌ Error terjadi: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
