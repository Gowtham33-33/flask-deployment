from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load trained model with error handling
model_path = 'iris_model.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found. Ensure the file exists.")

try:
    model = joblib.load(model_path)
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form[f'feature{i}']) for i in range(1, 5)]
    except ValueError:
        return render_template('result.html', prediction="Invalid input. Please enter numeric values.")

    try:
        prediction_index = model.predict([features])[0]
        class_names = ['Setosa', 'Versicolor', 'Virginica']
        if prediction_index not in [0, 1, 2]:
            raise ValueError("Model prediction out of expected range.")
        result = class_names[prediction_index]
    except Exception as e:
        result = f"Error making prediction: {e}"

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=False)  # Turn off debug in production


