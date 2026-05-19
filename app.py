from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load model
model = joblib.load('model_triage.pkl')

# Route baru untuk memunculkan halaman Dashboard HTML
@app.route('/')
def home():
    return render_template('index.html')

# Route untuk memproses prediksi
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = pd.DataFrame([{
        'temperature': data['temperature'],
        'heartrate': data['heartrate'],
        'resprate': data['resprate'],
        'o2sat': data['o2sat'],
        'sbp': data['sbp'],
        'dbp': data['dbp'],
        'pain': data['pain']
    }])
    
    prediction = model.predict(features)[0]
    return jsonify({'acuity': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)