from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import json
import numpy as np

app = Flask(__name__)
CORS(app)  # Enables Cross-Origin Resource Sharing

# Load model and data
__locations = None
__data_columns = None
__model = None

def load_saved_artifacts():
    print("Loading saved artifacts...start")
    global __locations
    global __data_columns
    global __model

    with open("artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]  # First 3 columns are sqft, bath, bhk

    with open("artifacts/banglore_home_prices_model.pickle", "rb") as f:
        __model = pickle.load(f)
    print("Loading saved artifacts...done")

def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except ValueError:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)

@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    response = jsonify({
        'locations': __locations
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    data = request.get_json()
    print("Received Data:", data)  # Debugging line

    try:
        total_sqft = float(data['total_sqft'])
        bhk = int(data['bhk'])
        bath = int(data['bath'])
        location = data['location']

        # Ensure location is valid
        if location not in __locations:
            return jsonify({'error': 'Invalid location'}), 400

        estimated_price = get_estimated_price(location, total_sqft, bhk, bath)
        return jsonify({'estimated_price': estimated_price})
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': 'Invalid input'}), 400

if __name__ == "__main__":
    print("Starting Flask Server for Home Price Prediction...")
    load_saved_artifacts()
    app.run(debug=True)

