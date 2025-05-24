import numpy as np
import joblib

# Load saved model and scaler
best_model = joblib.load('best_xgb_parkinson_model.joblib')
scaler = joblib.load('scaler.joblib')

# New input data with the exact 4 features used for training
new_data = np.array([[3.5, 10.2, 61.0, 41.4]])  # example values

# Scale new data using the loaded scaler
new_data_scaled = scaler.transform(new_data)

# Predict Parkinson Disorder Probability
prediction = best_model.predict(new_data_scaled)

print(f"Predicted Parkinson Disorder Probability: {prediction[0]:.2f}%")
