import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import requests
import plotly.graph_objects as go
from datetime import datetime
import time

# Page config
st.set_page_config(
    page_title="Mechanical Marvels - Crane Monitoring",
    page_icon="🏗️",
    layout="wide"
)

# ThingSpeak Configuration
THINGSPEAK_CHANNEL_ID = "2887938"
THINGSPEAK_API_KEY = "JILXVPKLMUFG8N2G"
THINGSPEAK_URL = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds.json?api_key={THINGSPEAK_API_KEY}&results=20"

# Status mapping for predictions
status_mapping = {
    0: {"status": "All is well", "color": "green"},
    1: {"status": "Medium risk", "color": "yellow"},
    2: {"status": "High Danger", "color": "red"},
}

@st.cache_data
def load_data():
    df = pd.read_csv("rope_ml_training_dataset.csv")
    return df

@st.cache_resource
def train_model(df):
    X = df.drop(columns=["Prediction"])
    y = df["Prediction"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return rf_model, accuracy

def fetch_thingspeak_data():
    try:
        response = requests.get(THINGSPEAK_URL)
        data = response.json()
        return data['feeds']
    except:
        return None

def main():
    st.title("🏗️ Mechanical Marvels - Crane Monitoring System")
    
    # Initialize ML model
    df = load_data()
    model, accuracy = train_model(df)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Dashboard", "ML Predictions"])
    
    if page == "Dashboard":
        st.header("Real-time IoT Monitoring Dashboard")
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        # Fetch latest data
        sensor_data = fetch_thingspeak_data()
        
        if sensor_data and len(sensor_data) > 0:
            latest = sensor_data[-1]
            
            with col1:
                st.metric("Oil Level", f"{float(latest['field1']):.2f}%")
                st.metric("Temperature", f"{float(latest['field2']):.1f}°C")
            
            with col2:
                st.metric("Wear Level", f"{float(latest['field3']):.2f}%")
                st.metric("Current", f"{float(latest['field7']):.2f}A")
            
            with col3:
                st.metric("Vibration X", f"{float(latest['field4']):.3f}g")
                st.metric("Vibration Y", f"{float(latest['field5']):.3f}g")
                st.metric("Vibration Z", f"{float(latest['field6']):.3f}g")
            
            # Create time series plots
            st.subheader("Historical Data")
            
            # Convert data for plotting
            df_sensor = pd.DataFrame(sensor_data)
            df_sensor['created_at'] = pd.to_datetime(df_sensor['created_at'])
            
            # Plot using Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_sensor['created_at'], y=df_sensor['field1'], name='Oil Level'))
            fig.add_trace(go.Scatter(x=df_sensor['created_at'], y=df_sensor['field2'], name='Temperature'))
            fig.update_layout(
                title='Oil Level and Temperature Over Time',
                xaxis_title='Time',
                yaxis_title='Value',
                template='plotly_dark'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Vibration Analysis
            fig_vib = go.Figure()
            fig_vib.add_trace(go.Scatter(x=df_sensor['created_at'], y=df_sensor['field4'], name='X-axis'))
            fig_vib.add_trace(go.Scatter(x=df_sensor['created_at'], y=df_sensor['field5'], name='Y-axis'))
            fig_vib.add_trace(go.Scatter(x=df_sensor['created_at'], y=df_sensor['field6'], name='Z-axis'))
            fig_vib.update_layout(
                title='Vibration Analysis',
                xaxis_title='Time',
                yaxis_title='Vibration (g)',
                template='plotly_dark'
            )
            st.plotly_chart(fig_vib, use_container_width=True)
            
            # Auto-refresh
            time.sleep(5)
            st.experimental_rerun()
            
        else:
            st.error("Unable to fetch sensor data. Please check your connection.")
    
    else:  # ML Predictions page
        st.header("Maintenance Prediction System")
        st.info(f"Model Accuracy: {accuracy*100:.2f}%")
        
        # Input form for predictions
        with st.form("prediction_form"):
            st.subheader("Enter Sensor Values")
            col1, col2 = st.columns(2)
            
            with col1:
                feature_1 = st.number_input("Oil Level (%)", 0.0, 100.0, 50.0)
                feature_2 = st.number_input("Temperature (°C)", 0.0, 150.0, 25.0)
            
            with col2:
                feature_3 = st.number_input("Vibration (g)", 0.0, 10.0, 1.0)
                feature_4 = st.number_input("Current (A)", 0.0, 50.0, 10.0)
            
            submitted = st.form_submit_button("Predict")
            
            if submitted:
                features = [feature_1, feature_2, feature_3, feature_4]
                prediction = model.predict([features])[0]
                status = status_mapping[prediction]
                
                st.markdown(f"""
                <div style='padding: 20px; border-radius: 10px; background-color: {status['color']}20;
                border: 2px solid {status['color']}'>
                <h3 style='color: {status['color']}'>{status['status']}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Show feature importance
                if st.checkbox("Show Feature Importance"):
                    feature_importance = pd.DataFrame({
                        'Feature': ['Oil Level', 'Temperature', 'Vibration', 'Current'],
                        'Importance': model.feature_importances_
                    })
                    feature_importance = feature_importance.sort_values('Importance', ascending=False)
                    
                    fig = go.Figure(go.Bar(
                        x=feature_importance['Feature'],
                        y=feature_importance['Importance'],
                        marker_color='lightblue'
                    ))
                    fig.update_layout(
                        title='Feature Importance',
                        xaxis_title='Feature',
                        yaxis_title='Importance',
                        template='plotly_dark'
                    )
                    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
