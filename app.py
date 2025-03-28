import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# 🔥 Load dataset with error handling
@st.cache_data
def load_data(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df
    else:
        st.error(f"❌ CSV file not found: `{file_path}`")
        return pd.DataFrame()  # Return empty DataFrame if file not found

# 🔥 Train the model
@st.cache_data
def train_model(df):
    if df.empty:
        return None, 0.0

    X = df.drop(columns=["Prediction"])
    y = df["Prediction"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return rf_model, accuracy

# 🔥 Status mapping
status_mapping = {
    0: {"status": "✅ All is well", "color": "#28a745", "emoji": "🟢"},
    1: {"status": "⚠️ Medium risk", "color": "#ffc107", "emoji": "🟡"},
    2: {"status": "🚨 High Danger", "color": "#dc3545", "emoji": "🔴"},
}

# 🔥 Main function
def main():
    st.title("🛠️ Rope Safety Prediction System")

    # CSV File Selection
    uploaded_file = st.file_uploader("📄 Upload CSV file", type=["csv"])
    
    if uploaded_file:
        file_path = "uploaded_file.csv"
        
        # Save uploaded file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        df = load_data(file_path)
        
        # Check if dataset is valid
        if not df.empty and "Prediction" in df.columns:
            model, accuracy = train_model(df)

            if model:
                st.sidebar.header("📊 Model Accuracy")
                st.sidebar.success(f"Accuracy: {accuracy * 100:.2f}%")

                st.sidebar.header("🛠️ Input Features")
                feature_1 = st.sidebar.number_input("Feature 1", min_value=0.0, max_value=100.0, step=0.1)
                feature_2 = st.sidebar.number_input("Feature 2", min_value=0.0, max_value=100.0, step=0.1)
                feature_3 = st.sidebar.number_input("Feature 3", min_value=0.0, max_value=100.0, step=0.1)
                feature_4 = st.sidebar.number_input("Feature 4", min_value=0.0, max_value=100.0, step=0.1)

                if st.sidebar.button("🔍 Predict"):
                    features = np.array([[feature_1, feature_2, feature_3, feature_4]])
                    prediction = model.predict(features)[0]
                    response = status_mapping[prediction]

                    st.markdown(f"### {response['emoji']} Prediction Result: **{response['status']}**")

                    st.markdown(
                        f"<div style='background-color:{response['color']}; padding:15px; border-radius:10px;'>"
                        f"<h2 style='color:white;'>Status: {response['status']}</h2>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

            else:
                st.error("⚠️ Model training failed.")
        else:
            st.error("❌ Invalid CSV format. Make sure it contains the 'Prediction' column.")
    else:
        st.info("📤 Please upload a CSV file to proceed.")


if __name__ == "__main__":
    main()
