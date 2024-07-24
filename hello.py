import pandas as pd
from prophet import Prophet
import streamlit as st

# Load your stock price data
# The data should be a CSV file with columns 'Date' and the column you want to predict
data = pd.read_csv('ZOMATO.NS.csv')

# Prepare the data for Prophet
# Prophet expects the columns to be named 'ds' and 'y'
data['ds'] = pd.to_datetime(data['Date'])

# Streamlit text input for the column to predict
m = st.text_input("Enter the column name to predict:")

# Proceed only if a valid input is given
if m and m in data.columns:
    data['y'] = data[m]
    data = data[['ds', 'y']]
    
    # Split the data into training and testing sets
    train_data = data.iloc[:-365]  # all data except the last 365 days
    test_data = data.iloc[-365:]  # last 365 days

    # Initialize the Prophet model
    model = Prophet()
    
    # Fit the model
    model.fit(train_data)
    
    # Create a DataFrame with future dates
    future = model.make_future_dataframe(periods=365)
    
    # Make predictions
    forecast = model.predict(future)
    
    # Display the forecast
    st.write(forecast)
else:
    st.write("Please enter a valid column name from the dataset.")
