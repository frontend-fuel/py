import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# 🔥 CSS Styling for Website-Like Design
st.markdown("""
    <style>
        .main-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px;
            background: #f9f9f9;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            border-radius: 10px;
        }
        .header {
            color: white;
            background: #007BFF;
            padding: 20px;
            text-align: center;
            border-radius: 10px 10px 0 0;
        }
        .footer {
            text-align: center;
            margin-top: 20px;
            color: #555;
        }
        .btn-predict {
            background-color: #28a745;
            color: white;
            border: none;
            padding: 10px 25px;
            border-radius: 5px;
            cursor: pointer;
            transition: 0.3s;
        }
        .btn-predict:hover {
            background-color: #218838;
        }
    </style>
""", unsafe_allow_html=True)

# 🔥 Load Dataset with Error Handling
@st.cache_data
def load_data(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df
    else:
        st.error(f"❌ CSV file not found: `{file_path}`")
        return pd.DataFrame()

# 🔥 Train Model
@st.cache_data
def train_model(df):
    if df.empty:
        return None, 0.0

    X = df.drop(columns=["Prediction"])
    y = df["Prediction"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy

# 🔥 Status Mapping
status_mapping = {
    0: {"status": "✅ All is well", "color": "#28a745", "emoji": "🟢"},
    1: {"status": "⚠️ Medium risk", "color": "#ffc107", "emoji": "🟡"},
    2: {"status": "🚨 High Danger", "color": "#dc3545", "emoji": "🔴"},
}

# 🔥 Main Function
def main():
    st.markdown("<div class='header'><h1>🛠️ Rope Safety Prediction System</h1></div>", unsafe_allow_html=True)

    st.markdown("<div class='main-container'>", unsafe_allow_html=True)

    # Upload CSV File
    uploaded_file = st.file_uploader("📄 Upload CSV file", type=["csv"])
    
    if uploaded_file:
        file_path = "uploaded_file.csv"
        
        # Save uploaded file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        df = load_data(file_path)

        if not df.empty and "Prediction" in df.columns:
            model, accuracy = train_model(df)

            if model:
                st.success(f"✅ Model trained successfully with {accuracy * 100:.2f}% accuracy!")

                # 🔥 User Input for Features
                st.subheader("🔍 Enter Features:")
                col1, col2, col3, col4 = st.columns(4)
                feature_1 = col1.number_input("Feature 1", min_value=0.0, max_value=100.0, step=0.1)
                feature_2 = col2.number_input("Feature 2", min_value=0.0, max_value=100.0, step=0.1)
                feature_3 = col3.number_input("Feature 3", min_value=0.0, max_value=100.0, step=0.1)
                feature_4 = col4.number_input("Feature 4", min_value=0.0, max_value=100.0, step=0.1)

                if st.button("🔍 Predict", key="predict_btn"):
                    features = np.array([[feature_1, feature_2, feature_3, feature_4]])
                    prediction = model.predict(features)[0]
                    response = status_mapping[prediction]

                    # Display Prediction Result
                    st.markdown(f"""
                        <div style='background-color:{response['color']}; padding:20px; border-radius:10px; text-align:center;'>
                            <h2 style='color:white;'>Prediction: {response['status']}</h2>
                            <h3 style='color:white;'>{response['emoji']} Safety Level</h3>
                        </div>
                    """, unsafe_allow_html=True)

            else:
                st.error("⚠️ Model training failed.")
        else:
            st.error("❌ Invalid CSV format. Make sure it contains the 'Prediction' column.")
    else:
        st.info("📤 Please upload a CSV file to proceed.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='footer'>🚀 Rope Safety Prediction System © 2025</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
