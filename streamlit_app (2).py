import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler

# Load the LSTM model
@st.cache_resource
def load_lstm_model():
    return tf.keras.models.load_model('lstm_model.keras')

# Load the StandardScaler
@st.cache_resource
def load_scaler():
    with open('standard_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return scaler

# Optimized data loading
@st.cache_data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

lstm_model = load_lstm_model()
scaler = load_scaler()
n_steps = 30 

st.title('AAPL Stock Price Prediction (LSTM Model)')
st.markdown("Forecast Apple (AAPL) stock prices using a trained Long Short-Term Memory neural network.")

st.header("1. Historical Data")

try:
    historical_data_path = 'AAPL (5).csv'
    original_data = load_and_preprocess_data(historical_data_path)
    
    # CRITICAL FIX: Use .transform() instead of .fit_transform() to avoid recalculating mean/std
    original_data['Adj Close Scaled'] = scaler.transform(original_data[['Adj Close']])
    data_for_plot = original_data.rename(columns={'Adj Close Scaled': 'Adj Close_Scaled'})

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=data_for_plot['Date'], y=data_for_plot['Adj Close_Scaled'], mode='lines', name='Historical Scaled Price'))
    fig_hist.update_layout(title='Historical Adjusted Close Price (Scaled)', xaxis_title='Date', yaxis_title='Scaled Price')
    st.plotly_chart(fig_hist)

    st.write("Last 5 historical records:")
    st.dataframe(original_data[['Date', 'Adj Close']].tail())

except FileNotFoundError:
    st.error(f"File '{historical_data_path}' not found. Please upload it to the directory.")
    st.stop()

st.header("2. Predict Future Prices")
forecast_days = st.slider("Select number of days to forecast:", 1, 60, 7)

if st.button('Generate Forecast'):
    # Prepare the most recent sequence for prediction
    last_n_steps_data = list(data_for_plot['Adj Close_Scaled'].iloc[-n_steps:].values)

    future_forecasts = []
    future_dates = []
    last_historical_date = data_for_plot['Date'].iloc[-1]

    for i in range(forecast_days):
        # Reshape for LSTM input: (samples, time_steps, features)
        current_input = np.array(last_n_steps_data[-n_steps:]).reshape(1, n_steps, 1)
        
        # Predict and update sequence
        predicted_scaled_value = lstm_model.predict(current_input, verbose=0)[0, 0]
        future_forecasts.append(predicted_scaled_value)
        last_n_steps_data.append(predicted_scaled_value)

        # Generate business day dates
        next_date = last_historical_date + pd.Timedelta(days=1)
        while next_date.dayofweek > 4: # Skip Saturdays and Sundays
            next_date += pd.Timedelta(days=1)
        future_dates.append(next_date)
        last_historical_date = next_date

    # Inverse transform to get actual currency values
    forecast_actual_prices = scaler.inverse_transform(np.array(future_forecasts).reshape(-1, 1))

    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecasted Price': forecast_actual_prices.flatten()
    })

    st.subheader("Forecasted Results")
    st.dataframe(forecast_df)

    # Combined Plot
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=data_for_plot['Date'], y=data_for_plot['Adj Close_Scaled'], mode='lines', name='Historical (Scaled)'))
    fig_forecast.add_trace(go.Scatter(x=future_dates, y=future_forecasts, mode='lines+markers', name='Forecast (Scaled)', line=dict(color='red', dash='dot')))
    fig_forecast.update_layout(title=f'{forecast_days}-Day Stock Price Forecast', xaxis_title='Date', yaxis_title='Scaled Value')
    st.plotly_chart(fig_forecast)

st.divider()
st.caption("**Disclaimer:** Financial markets are volatile. This tool is for informational purposes only.")
