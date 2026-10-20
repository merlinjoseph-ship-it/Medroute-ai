import os
from flask import Flask, jsonify, render_template, request
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load datasets
PATIENT_FILE = "patients.csv"
HOSPITAL_FILE = "Hospitals.csv"

# The source CSV files are tab-delimited; reading them with the default comma
# delimiter produces a single malformed column and breaks the app.
df_patients = pd.read_csv(PATIENT_FILE, sep="\t")
df_hospitals = pd.read_csv(HOSPITAL_FILE, sep="\t")

# Normalize all column names to avoid issues caused by stray whitespace.
df_patients.columns = [str(col).strip() for col in df_patients.columns]
df_hospitals.columns = [str(col).strip() for col in df_hospitals.columns]

# Ensure the hospital coordinate columns are numeric.
for col in ["Latitude", "Longitude", "ICU Beds", "Emergency Beds"]:
    if col in df_hospitals.columns:
        df_hospitals[col] = pd.to_numeric(df_hospitals[col], errors="coerce")

# Encode categorical feature for ML model
le_emergency = LabelEncoder()
df_patients["Emergency_Type_Enc"] = le_emergency.fit_transform(
    df_patients["Emergency Type"].astype(str).str.strip()
)

# Train Random Forest Model on Startup
X = df_patients[
    ["Age", "Heart Rate", "Blood Pressure", "SpO2", "Emergency_Type_Enc"]
]
y = df_patients["Severity"]

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)


def calculate_distance(lat1, lon1, lat2, lon2):
  """Haversine formula to calculate straight-line distance in km."""
  R = 6371
  dlat = np.radians(lat2 - lat1)
  dlon = np.radians(lon2 - lon1)
  a = (
      np.sin(dlat / 2) ** 2
      + np.cos(np.radians(lat1))
      * np.cos(np.radians(lat2))
      * np.sin(dlon / 2) ** 2
  )
  c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
  return R * c


def get_real_road_eta(lat1, lon1, lat2, lon2, road_condition="Normal"):
  """Calculates real road distance and traffic/condition-adjusted ETA via OSRM."""
  try:
    # Query OSRM Public Driving API
    url = f"https://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"
    response = requests.get(url, timeout=4)
    data = response.json()

    if data.get("code") == "Ok" and len(data["routes"]) > 0:
      route = data["routes"][0]
      road_distance_km = round(route["distance"] / 1000.0, 2)
      base_duration_min = round(route["duration"] / 60.0, 1)
    else:
      raise ValueError("OSRM route not found")
  except Exception:
    # Fallback to Haversine with urban road winding coefficient (1.35x)
    road_distance_km = round(calculate_distance(lat1, lon1, lat2, lon2) * 1.35, 2)
    base_duration_min = round((road_distance_km / 40.0) * 60.0, 1)

  # Road condition & Traffic multipliers
  multipliers = {
      "Optimal": 0.85,  # Clear midnight highway / express transit
      "Normal": 1.00,  # Standard daytime flow
      "Moderate": 1.25,  # Moderate suburban traffic / curves
      "Heavy": 1.55,  # Peak-hour gridlock / monsoon rain
  }
  factor = multipliers.get(road_condition, 1.0)
  predicted_eta_min = round(base_duration_min * factor, 1)

  return road_distance_km, predicted_eta_min


@app.route("/")
def home():
  return render_template("home.html")


@app.route("/api/search_location")
def search_location():
  """Autocomplete search for users without live GPS access."""
  query = request.args.get("q", "").strip()
  if len(query) < 2:
    return jsonify([])

  try:
    # Use OpenStreetMap Nominatim with Kerala/India bias
    url = "https://nominatim.openstreetmap.org/search"
    headers = {"User-Agent": "MedRouteEmergencyApp/2.0"}
    params = {
        "q": query,
        "format": "json",
        "countrycodes": "in",
        "limit": 5,
        "addressdetails": 1,
    }
    res = requests.get(url, params=params, headers=headers, timeout=5)
    return jsonify(res.json())
  except Exception as e:
    return jsonify({"error": str(e)}), 500


@app.route("/dispatch", methods=["GET", "POST"])
def dispatch():
  prediction = None
  recommended_hospital = None
  hospital_details = None
  user_lat, user_lon = 10.5276, 76.2144  # Fallback: Thrissur center
  location_name = "Default Coordinates (Thrissur)"
  road_condition = "Normal"

  if request.method == "POST":
    try:
      # Capture coordinates (from GPS or manual search picker)
      req_lat = request.form.get("user_lat")
      req_lon = request.form.get("user_lon")
      if req_lat and req_lon:
        user_lat = float(req_lat)
        user_lon = float(req_lon)

      location_name = request.form.get(
          "location_name", f"{user_lat:.4f}, {user_lon:.4f}"
      )
      road_condition = request.form.get("road_condition", "Normal")

      age = float(request.form["age"])
      hr = float(request.form["heart_rate"])
      bp = float(request.form["blood_pressure"])
      spo2 = float(request.form["spo2"])
      emergency_type = request.form["emergency_type"].strip()

      et_enc = (
          le_emergency.transform([emergency_type])[0]
          if emergency_type in le_emergency.classes_
          else 0
      )
      input_data = np.array([[age, hr, bp, spo2, et_enc]])

      # Predict Emergency Severity
      prediction = rf_model.predict(input_data)[0]

      # Hospital Filtering logic
      filtered_hospitals = df_hospitals.copy()
      if prediction == "High":
        filtered_hospitals = filtered_hospitals[
            (filtered_hospitals["Trauma Facility"] == "Yes")
            & (filtered_hospitals["ICU Beds"] > 0)
        ]
      elif prediction == "Medium":
        filtered_hospitals = filtered_hospitals[
            filtered_hospitals["Emergency Beds"] > 0
        ]

      if filtered_hospitals.empty:
        filtered_hospitals = df_hospitals

      # Candidate selection using proximity
      filtered_hospitals["Haversine_km"] = filtered_hospitals.apply(
          lambda row: calculate_distance(
              user_lat, user_lon, row["Latitude"], row["Longitude"]
          ),
          axis=1,
      )
      best_hospital = filtered_hospitals.loc[
          filtered_hospitals["Haversine_km"].idxmin()
      ]

      hosp_lat = float(best_hospital["Latitude"])
      hosp_lon = float(best_hospital["Longitude"])

      # Calculate accurate road distance and condition-adjusted ETA
      actual_road_km, actual_eta_min = get_real_road_eta(
          user_lat, user_lon, hosp_lat, hosp_lon, road_condition
      )

      recommended_hospital = best_hospital["Hospital"]
      hospital_details = {
          "lat": hosp_lat,
          "lon": hosp_lon,
          "icu": int(best_hospital["ICU Beds"]),
          "emergency_beds": int(best_hospital["Emergency Beds"]),
          "trauma": best_hospital["Trauma Facility"],
          "distance": actual_road_km,
          "eta": actual_eta_min,
          "condition": road_condition,
      }

    except Exception as e:
      prediction = f"Error processing triage request: {str(e)}"

  emergency_types = list(le_emergency.classes_)

  return render_template(
      "index.html",
      prediction=prediction,
      hospital=recommended_hospital,
      details=hospital_details,
      user_lat=user_lat,
      user_lon=user_lon,
      location_name=location_name,
      emergency_types=emergency_types,
  )


if __name__ == "__main__":
  import os

  port = int(os.environ.get("PORT", 5000))
  app.run(host="0.0.0.0", port=port)