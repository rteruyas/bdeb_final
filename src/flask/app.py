from flask import Flask, request, jsonify
from flask_swagger_ui import get_swaggerui_blueprint
import pickle
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Construct the model path relative to the working directory (/app)
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Define a simple root route to confirm the API is running.
@app.route("/")
def index():
    return "API is running!"

# Configure Swagger UI.
SWAGGER_URL = '/api/docs'
API_URL = '/static/swagger.yaml'
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={'app_name': "Model Prediction API"}
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        columns = payload.get("columns")
        data = payload.get("data")
        
        if not columns or not data:
            return jsonify({"error": "Both 'columns' and 'data' keys are required."}), 400
        
        # Convert data into a DataFrame
        input_df = pd.DataFrame(data, columns=columns)
        
        # Get predictions
        predictions = model.predict(input_df)
        
        return jsonify({"predictions": predictions.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("Starting API server...")
    app.run(debug=True, host="0.0.0.0", port=5000)
