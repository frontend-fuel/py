import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("rope_ml_training_dataset.csv")  # Ensure CSV file is in the same directory
    return df

# Train the model
@st.cache_data
def train_model(df):
    X = df.drop(columns=["Prediction"])
    y = df["Prediction"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return rf_model, accuracy

# Status mapping
status_mapping = {
    0: {"status": "All is well", "color": "Green", "bg_color": "#28a745"},
    1: {"status": "Medium risk", "color": "Yellow", "bg_color": "#ffc107"},
    2: {"status": "High Danger", "color": "Red", "bg_color": "#dc3545"},
}

# Main function
def main():
    st.title("🛠️ Rope Safety Prediction System")
    
    # Load and train the model
    df = load_data()
    model, accuracy = train_model(df)
    
    st.sidebar.header("⚙️ Model Accuracy")
    st.sidebar.success(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    st.sidebar.header("📊 Input Features")
    
    feature_1 = st.sidebar.number_input("Feature 1", min_value=0.0, max_value=100.0, step=0.1)
    feature_2 = st.sidebar.number_input("Feature 2", min_value=0.0, max_value=100.0, step=0.1)
    feature_3 = st.sidebar.number_input("Feature 3", min_value=0.0, max_value=100.0, step=0.1)
    feature_4 = st.sidebar.number_input("Feature 4", min_value=0.0, max_value=100.0, step=0.1)

    if st.sidebar.button("🔍 Predict"):
        features = np.array([[feature_1, feature_2, feature_3, feature_4]])
        prediction = model.predict(features)[0]
        response = status_mapping[prediction]

        st.markdown(f"### 🛡️ Prediction Result: **{response['status']}**")
        
        st.markdown(
            f"<div style='background-color:{response['bg_color']};padding:10px;border-radius:10px;'>"
            f"<h2 style='color:white;'>Status: {response['status']}</h2>"
            f"</div>",
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
