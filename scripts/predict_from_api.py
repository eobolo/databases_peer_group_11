import requests

# Hosted API endpoints (from your Render deployment)
BASE_URL = "https://databases-and-apis.onrender.com"
POSTGRES_URL = f"{BASE_URL}/individuals/"
MONGO_URL = f"{BASE_URL}/mongo/individuals/"
PREDICT_URL = f"{BASE_URL}/predict"

# Available options
AVAILABLE_MODELS = ["Model 1", "Model 2", "Model 3"]
AVAILABLE_DBS = ["PostgreSQL", "MongoDB"]

# User input
print("Available models:", ", ".join(AVAILABLE_MODELS))
model_name = input("Enter the model to use (e.g., Model 1): ").strip()
if model_name not in AVAILABLE_MODELS:
    print(f"Error: Invalid model. Choose from {AVAILABLE_MODELS}")
    exit(1)

print("Available databases:", ", ".join(AVAILABLE_DBS))
db_source = input("Enter the database source (PostgreSQL or MongoDB): ").strip()
if db_source not in AVAILABLE_DBS:
    print(f"Error: Invalid database. Choose from {AVAILABLE_DBS}")
    exit(1)

# Fetch the latest entry from the selected database
url = POSTGRES_URL if db_source == "PostgreSQL" else MONGO_URL
params = {
    "limit": 1,
    "order_by": "individual_id desc" if db_source == "PostgreSQL" else "individual_id desc"
}
try:
    response = requests.get(url, params=params)
    response.raise_for_status()  # Raises an exception for 4xx/5xx status codes
    data = response.json()
    if not data:
        print(f"Error: No entries found in {db_source}")
        exit(1)
    latest_entry = data[0]
    print(f"Fetched latest entry from {db_source}:", latest_entry)
except requests.RequestException as e:
    print(f"Error: Failed to fetch data from {db_source} - {str(e)}")
    exit(1)

# Prepare data for /predict endpoint
predict_data = {
    "age": latest_entry["age"],
    "fnlwgt": latest_entry["fnlwgt"],
    "educational_num": latest_entry["educational_num"],
    "capital_gain": latest_entry.get("capital_gain", 0),
    "capital_loss": latest_entry.get("capital_loss", 0),
    "hours_per_week": latest_entry.get("hours_per_week", 40),
    "gender": latest_entry["gender"],
    "workclass": latest_entry["workclass"],
    "education": latest_entry["education"],
    "marital_status": latest_entry["marital_status"],
    "occupation": latest_entry["occupation"],
    "relationship": latest_entry["relationship"],
    "race": latest_entry["race"],
    "country": latest_entry["country"]
}

# Send to /predict endpoint
try:
    predict_params = {"model_name": model_name}
    response = requests.post(PREDICT_URL, json=predict_data, params=predict_params)
    response.raise_for_status()
    prediction = response.json()
    print(f"\nPrediction Results using {model_name}:")
    print("Model Used:", prediction["model_used"])
    print(f"Prediction Probability: {prediction['prediction_probability']:.4f}")
    print("Prediction Class:", prediction["prediction_class"])
    print("Income Prediction:", prediction["income_prediction"])
except requests.RequestException as e:
    print(f"Error: Failed to get prediction - {str(e)}")
    exit(1)