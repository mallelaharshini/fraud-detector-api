from flask import Flask, request, jsonify
import joblib  # ✅ Use joblib instead of pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('models/fraud_model.pkl')  # ✅ Correct loading

@app.route('/')
def home():
    return "Fraud Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
