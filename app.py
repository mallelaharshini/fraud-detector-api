from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("models/fraud_model.pkl")

@app.route('/')
def home():
    return "Fraud Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON input
        data = request.get_json(force=True)

        # Convert input into NumPy array for prediction
        input_data = np.array(data['features']).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Return result
        result = "Fraud" if prediction == 1 else "Not Fraud"
        return jsonify({"prediction": int(prediction), "result": result})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
