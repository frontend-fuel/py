import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 🔥 Full-Screen and Modern Styling
st.markdown("""
    <style>
        /* Full-screen layout */
        body, .main-container {
            margin: 0;
            padding: 0;
            height: 100vh;
            width: 100vw;
            background: linear-gradient(135deg, #74EBD5, #ACB6E5);
            overflow: hidden;
            transition: background 0.5s ease-in-out;
        }
        .main-container {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            padding: 40px;
        }
        .header {
            color: white;
            text-align: center;
            padding: 30px;
            border-radius: 10px 10px 0 0;
            width: 90%;
            max-width: 1200px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.3);
            background: #007BFF;
            transition: all 0.5s;
        }
        .footer {
            text-align: center;
            color: #fff;
            margin-top: 20px;
            font-size: 14px;
            opacity: 0.8;
        }
        .btn-predict {
            background: #28a745;
            color: white;
            border: none;
            padding: 12px 30px;
            font-size: 18px;
            border-radius: 25px;
            cursor: pointer;
            transition: transform 0.3s ease, background 0.3s;
        }
        .btn-predict:hover {
            background: #218838;
            transform: scale(1.1);
        }
        .card {
            background: white;
            border-radius: 15px;
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.3);
            padding: 30px;
            width: 90%;
            max-width: 1200px;
            transition: transform 0.5s;
        }
        .card:hover {
            transform: scale(1.03);
        }
        .prediction-box {
            color: white;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            transition: all 0.5s;
        }
    </style>
""", unsafe_allow_html=True)

# 🔥 Load Dataset from GitHub
@st.cache_data
def load_data_from_github(url):
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"❌ Failed to load data from GitHub: {e}")
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
    0: {"status": "✅ All is well", "color": "#28a745", "emoji": "🟢", "bg": "#d4edda"},
    1: {"status": "⚠️ Medium risk", "color": "#ffc107", "emoji": "🟡", "bg": "#fff3cd"},
    2: {"status": "🚨 High Danger", "color": "#dc3545", "emoji": "🔴", "bg": "#f8d7da"},
}

# 🔥 Main Function
def main():
    st.markdown("<div class='header'><h1>🛠️ Rope Safety Prediction System</h1></div>", unsafe_allow_html=True)

    # 🔥 GitHub CSV URL
    github_url = "https://raw.githubusercontent.com/frontend-fuel/py/main/rope_ml_training_dataset.csv"

    # Load dataset
    df = load_data_from_github(github_url)

    if not df.empty and "Prediction" in df.columns:
        model, accuracy = train_model(df)

        if model:
            st.success(f"✅ Model trained successfully with {accuracy * 100:.2f}% accuracy!")

            # 🔥 User Input for Features
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("🔍 Enter Features:")
            col1, col2, col3, col4 = st.columns(4)
            feature_1 = col1.number_input("Feature 1", min_value=0.0, max_value=100.0, step=0.1)
            feature_2 = col2.number_input("Feature 2", min_value=0.0, max_value=100.0, step=0.1)
            feature_3 = col3.number_input("Feature 3", min_value=0.0, max_value=100.0, step=0.1)
            feature_4 = col4.number_input("Feature 4", min_value=0.0, max_value=100.0, step=0.1)

            st.markdown("</div>", unsafe_allow_html=True)

            if st.button("🔍 Predict", key="predict_btn"):
                features = np.array([[feature_1, feature_2, feature_3, feature_4]])
                prediction = model.predict(features)[0]
                response = status_mapping[prediction]

                # 🔥 Dynamic Background Color
                bg_color = response["bg"]
                st.markdown(
                    f"""
                    <style>
                        body, .main-container {{
                            background: {bg_color};
                        }}
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                # Display Prediction Result with Animation
                st.markdown(f"""
                    <div class='card prediction-box' style='background-color:{response['color']};'>
                        <h2 style='color:white;'>Prediction: {response['status']}</h2>
                        <h3 style='color:white;'>{response['emoji']} Safety Level</h3>
                    </div>
                """, unsafe_allow_html=True)

        else:
            st.error("⚠️ Model training failed.")
    else:
        st.error("❌ Invalid CSV format or GitHub URL.")

    st.markdown("<div class='footer'>🚀 Rope Safety Prediction System © 2025</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
