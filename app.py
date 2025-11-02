from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("diabetes_model.joblib")
scaler = joblib.load("diabetes_scaler.joblib")

@app.route('/')
def home():
    # Serve index.html from the templates folder
    from flask import render_template
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from frontend
        data = request.get_json()

        # Ensure all expected keys exist
        expected_keys = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]
        if not all(k in data for k in expected_keys):
            return jsonify({"error": "Missing fields in input"}), 400

        # Convert to numpy array
        features = np.array([[data[k] for k in expected_keys]])

        # Scale input
        scaled_input = scaler.transform(features)

        # Predict
        prediction = int(model.predict(scaled_input)[0])
        probability = float(model.predict_proba(scaled_input)[0][1])

        # Return as JSON
        return jsonify({
            "prediction": prediction,
            "diabetes_probability": probability
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
