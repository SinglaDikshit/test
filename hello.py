import streamlit as st
from prophet import Prophet
import pandas as pd

st.header("Test")

# Example usage of prophet
df = pd.read_csv("ZOMATO.NS.csv")
m = Prophet()
m.fit(df)
st.write("forecast")
