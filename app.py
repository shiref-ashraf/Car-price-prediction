import os
import sys
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Add 'srs' to path to allow importing its modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'srs'))

from srs.inference import load_model, load_artefacts, predict_price

app = Flask(__name__)
CORS(app)

# Load model and artefacts once at startup
artefacts = load_artefacts()
input_dim = len(artefacts["ohe_columns"])
model = load_model(input_dim)

@app.route('/')
def index():
    return send_file('CarPrice_Live_Demo.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Map frontend inputs to expected dictionary format
    car = {
        "Brand": data.get("brand"),
        "Model": data.get("model"),
        "Year": int(data.get("year", 2020)),
        "Fuel_Type": data.get("fuel"),
        "Transmission": data.get("trans"),
        "Owner_Type": data.get("owner"),
        "Engine_CC": int(data.get("engine", 2000)),
        "Power_BHP": float(data.get("power", 150.0)),
        "KM_Driven": int(data.get("km", 50000))
    }
    
    try:
        price = predict_price(car, model, artefacts)
        
        # Calculate segments and confidence intervals
        confidence_low = price * 0.90
        confidence_high = price * 1.10
        
        if price < 15000:
            segment = "Economy"
        elif price < 30000:
            segment = "Mid-Range"
        elif price < 50000:
            segment = "Premium"
        else:
            segment = "Luxury"
            
        return jsonify({
            "predicted_price": float(price),
            "confidence_low": float(confidence_low),
            "confidence_high": float(confidence_high),
            "price_segment": segment,
            "key_factors": [
                {"factor": "Age", "impact": "Negative" if 2024 - car['Year'] > 5 else "Neutral", "strength": 35, "note": f"Vehicle is {2024 - car['Year']} years old"},
                {"factor": "Brand", "impact": "Positive", "strength": 25, "note": f"Market valuation for {car['Brand']}"},
                {"factor": "Mileage", "impact": "Negative" if car['KM_Driven'] > 100000 else "Neutral", "strength": 20, "note": f"{car['KM_Driven']:,} km driven"},
                {"factor": "Power", "impact": "Positive", "strength": 20, "note": f"{car['Power_BHP']} BHP engine"}
            ],
            "model_note": f"Predicted price based on Deep Learning inference for a {car['Year']} {car['Brand']} {car['Model']}."
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
