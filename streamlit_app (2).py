import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os
from sklearn.preprocessing import StandardScaler

# Verified Plotly check
try:
    import plotly.graph_objects as go
except ModuleNotFoundError:
    st.error("Error: 'plotly' is not installed. Add 'plotly' to your requirements.txt file.")
    st.stop()

# Cache model and scaler for efficiency
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('lstm_model.keras')
    with open('standard_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# Load historical data
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

st.title('AAPL Stock Price Prediction')

# Asset Check
if not os.path.exists('lstm_model.keras') or not os.path.exists('standard_scaler.pkl'):
    st.error("Missing model or scaler files in the repository.")
    st.stop()

lstm_model, scaler = load_assets()
n_steps = 30

# Data Management
historical_path = 'AAPL (5).csv'
if os.path.exists(historical_path):
    data = load_data(historical_path)
    # Scale only (don't fit) to avoid data leakage
    data['Scaled'] = scaler.transform(data[['Adj Close']])
    
    # 1. Historical Chart
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=data['Date'], y=data['Adj Close'], name='Historical Price'))
    fig_hist.update_layout(title='AAPL History', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig_hist)
    
    # 2. Forecasting
    forecast_days = st.slider("Forecast Horizon (Days)", 1, 30, 7)
    if st.button('Predict'):
        last_window = list(data['Scaled'].iloc[-n_steps:].values)
        predictions = []
        future_dates = []
        last_date = data['Date'].iloc[-1]

        for _ in range(forecast_days):
            x_input = np.array(last_window[-n_steps:]).reshape(1, n_steps, 1)
            pred = lstm_model.predict(x_input, verbose=0)[0][0]
            predictions.append(pred)
            last_window.append(pred)
            
            # Add next business day
            last_date += pd.Timedelta(days=1)
            while last_date.dayofweek > 4: last_date += pd.Timedelta(days=1)
            future_dates.append(last_date)

        # Inverse transform to actual prices
        actual_preds = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        # Result Plot
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=future_dates, y=actual_preds.flatten(), name='Forecast', line=dict(color='red')))
        st.plotly_chart(fig_pred)
else:
    st.warning(f"Please upload {historical_path} to your repository.")

st.caption("Forecasts are for informational purposes only. Data source: Historical AAPL CSV.")

  
       
