import streamlit as st
from prophet import Prophet
import pandas as pd

st.header("Test")

# Example usage of prophet
df = pd.DataFrame({
    'ds': pd.date_range(start='2020-01-01', periods=30),
    'y': range(30)
})

m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=10)
forecast = m.predict(future)

st.write(forecast)
