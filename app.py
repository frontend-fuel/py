import threading
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load dataset
def load_data():
    df = pd.read_csv("rope_ml_training_dataset.csv")  # Ensure CSV file is in the same directory
    return df

# Train the model
def train_model(df):
    X = df.drop(columns=["Prediction"])
    y = df["Prediction"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return rf_model, accuracy

# Mapping predictions to background colors
status_mapping = {
    0: {"status": "All is well", "color": "Green", "bg_color": "#28a745"},
    1: {"status": "Medium risk", "color": "Yellow", "bg_color": "#ffc107"},
    2: {"status": "High Danger", "color": "Red", "bg_color": "#dc3545"},
}

df = load_data()
model, accuracy = train_model(df)

@app.route("/")
def home():
    return render_template('index1.html', accuracy=accuracy * 100)

@app.route("/predict", methods=["POST"])
def predict():
    features = [float(request.form[f"feature_{i}"]) for i in range(1, 5)]
    prediction = model.predict([features])[0]
    response = status_mapping[prediction]
    response["prediction"] = prediction
    return jsonify(response)

if __name__ == "__main__":
    app.run(port=5000, debug=True)