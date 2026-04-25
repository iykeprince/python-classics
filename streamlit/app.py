import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="IT Online Streamlit App", layout="centered")
st.title("Hello IT online Apps")
st.caption("Streamlit App Dashboard")
st.write("Hi This page is generate to showcase how Streamlit works")

st.subheader("Data captured from London Dataset")

df=pd.DataFrame({
    "city": ["Paris", "London", "Berlin", "Sweden"],
    "temp": [21, 24, 45,23],
    "raining": [False, False, True, False]
})
st.dataframe(df, width="content")

st.subheader("Line Chart for the data")

chart_df=pd.DataFrame(np.random.randn(20,5), columns=list("ABCVZ"))

st.line_chart(chart_df)
st.success("This is your app running")