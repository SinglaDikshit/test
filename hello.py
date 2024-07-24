import pandas as pd
from prophet import Prophet
import streamlit as st
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load your stock price data
# The data should be a CSV file with columns 'Date' and 'Close' (or adjust the column names accordingly)
data = pd.read_csv('ZOMATO.NS.csv')

# Prepare the data for Prophet
# Prophet expects the columns to be named 'ds' and 'y'
data['ds'] = pd.to_datetime(data['Date'])
m=st.text_input("What to predict:")
data['y'] = data[m]
data = data[['ds','y']]

train_data = data.iloc[:-365]
#    all data except the last 365 days

test_data = data.iloc[-365:] 
model = Prophet()
# Initialize the Prophet model
model = Prophet()

# Fit the model
model.fit(data)

# Create a DataFrame with future dates
# Here we predict for the next 365 days (1 year)
future = model.make_future_dataframe(periods=365)

# Make predictions
forecast = model.predict(future)
st.write(forecast)
