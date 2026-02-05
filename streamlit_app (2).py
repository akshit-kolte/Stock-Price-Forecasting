import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler # Ensure StandardScaler is imported

# Load the LSTM model
@st.cache_resource
def load_lstm_model():
    model = tf.keras.models.load_model('lstm_model.keras')
    return model

lstm_model = load_lstm_model()

# Load the StandardScaler
@st.cache_resource
def load_scaler():
    with open('standard_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return scaler

scaler = load_scaler()

# Assuming n_steps used during training
n_steps = 30 # This should match the n_steps used during model training

st.title('AAPL Stock Price Prediction (LSTM Model)')

st.markdown("""
This application forecasts Apple (AAPL) stock prices using a trained LSTM (Long Short-Term Memory) neural network model.
It visualizes historical adjusted close prices and provides predictions for future dates.
""")

st.header("1. Historical Data")

# Load the original data (ensure 'Date' and 'Adj Close' are present and preprocessed as during training)
# For simplicity, let's assume 'data' is available in its scaled form or load it from a source.
# In a real Streamlit app, you might load a CSV or connect to a database.

# Dummy data loading for demonstration, replace with actual data loading in a real scenario
# For this example, let's assume you have a 'data.csv' with 'Date' and 'Adj Close'
# and it has been preprocessed (scaled, etc.) similar to the notebook.
# If you save your processed data, you can load it here.

try:
    # Attempt to load the preprocessed data (if saved previously)
    # Or, if you have the original CSV, you'd load it and apply the scaler
    historical_data_path = 'AAPL (5).csv' # Assuming the original CSV is available
    original_data = pd.read_csv(historical_data_path)
    original_data['Date'] = pd.to_datetime(original_data['Date'])
    # Scale 'Adj Close' using the loaded scaler
    original_data['Adj Close Scaled'] = scaler.fit_transform(original_data[['Adj Close']])
    data_for_plot = original_data.rename(columns={'Adj Close Scaled': 'Adj Close'})

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=data_for_plot['Date'], y=data_for_plot['Adj Close'], mode='lines', name='Historical Adj Close (Scaled)'))
    fig_hist.update_layout(title='Historical Adjusted Close Price', xaxis_title='Date', yaxis_title='Adjusted Close Price (Scaled)')
    st.plotly_chart(fig_hist)

    st.write("Last 5 historical data points:")
    st.dataframe(original_data[['Date', 'Adj Close']].tail())

except FileNotFoundError:
    st.error("Historical data file 'AAPL (5).csv' not found. Please ensure it's in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading or processing historical data: {e}")
    st.stop()

st.header("2. Predict Future Prices")

# User input for number of forecast days
forecast_days = st.slider("Select number of days to forecast:", 1, 60, 7)

if st.button('Generate Forecast'):
    if 'Adj Close' not in data_for_plot.columns:
        st.error("Processed 'Adj Close' column not found for forecasting. Please check data loading.")
        st.stop()

    # Get the last n_steps 'Adj Close' values from the scaled historical data
    last_n_steps_data = list(data_for_plot['Adj Close'].iloc[-n_steps:].values)

    future_forecasts = []
    future_dates = []

    last_historical_date = data_for_plot['Date'].iloc[-1]

    for i in range(forecast_days):
        # Prepare the input sequence for the model
        current_input = np.array(last_n_steps_data[-n_steps:]).reshape(1, n_steps, 1)

        # Make prediction using the LSTM model
        predicted_scaled_value = lstm_model.predict(current_input, verbose=0)[0, 0]
        future_forecasts.append(predicted_scaled_value)

        # Update the input sequence with the new prediction
        last_n_steps_data.append(predicted_scaled_value)

        # Generate the next date (assuming daily business day frequency for forecasting)
        next_date = last_historical_date + pd.Timedelta(days=1)
        while next_date.dayofweek > 4: # Skip weekends
            next_date += pd.Timedelta(days=1)
        future_dates.append(next_date)
        last_historical_date = next_date

    # Create DataFrame for forecasts
    forecast_df_scaled = pd.DataFrame({
        'Date': future_dates,
        'Forecasted Adj Close (Scaled)': future_forecasts
    })

    # Inverse transform to get actual prices
    # The scaler was fitted on a single column 'Adj Close'. It expects a 2D array for inverse_transform.
    # So, reshape future_forecasts to (n_samples, 1)
    forecast_df_actual = forecast_df_scaled.copy()
    forecast_df_actual['Forecasted Adj Close'] = scaler.inverse_transform(np.array(future_forecasts).reshape(-1, 1))

    st.subheader("Forecasted Adjusted Close Prices")
    st.dataframe(forecast_df_actual[['Date', 'Forecasted Adj Close']])

    # Plot historical data and forecasts
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=data_for_plot['Date'], y=data_for_plot['Adj Close'], mode='lines', name='Historical Adj Close (Scaled)'))
    fig_forecast.add_trace(go.Scatter(x=forecast_df_scaled['Date'], y=forecast_df_scaled['Forecasted Adj Close (Scaled)'], mode='lines', name='Forecasted Adj Close (Scaled)', line=dict(color='red', dash='dot')))
    fig_forecast.update_layout(title=f'Historical and {forecast_days}-Day Forecast of Adjusted Close Price', xaxis_title='Date', yaxis_title='Adjusted Close Price (Scaled)')
    st.plotly_chart(fig_forecast)

    st.subheader("Disclaimer")
    st.write("Stock price prediction is inherently uncertain and past performance is not indicative of future results. This tool is for informational purposes only and should not be considered financial advice.")

