import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "features": [0.0, 0.1, -1.1, 2.2, -0.3, 1.2, 0.0, -0.5, 1.3, 0.2,
                 -0.1, 0.0, 1.0, 0.8, 0.1, -0.9, 0.4, -1.0, 0.6, 0.3,
                 0.0, 1.5, -0.8, 0.9, -0.2, 0.5, 0.3, 0.2, 0.1, 100.00]
}

response = requests.post(url, json=data)
print("Status Code:", response.status_code)
print("Response Text:", response.text)
