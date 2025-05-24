from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
best_model = joblib.load('best_xgb_parkinson_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Expecting a JSON with a 'features' key: [f1, f2, f3, f4]
    features = data.get('features')
    if not features or len(features) != 4:
        return jsonify({'error': 'Please provide 4 features.'}), 400
    arr = np.array([features])
    arr_scaled = scaler.transform(arr)
    prediction = best_model.predict(arr_scaled)
    # If prediction is probability, adjust as needed
    return jsonify({'prediction': float(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
