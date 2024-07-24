import streamlit as st
from prophet import Prophet
import pandas as pd

st.header("test")
data = pd.read_csv("ZOMATO.NS.csv")
model = Prophet()
model.fit(data)
