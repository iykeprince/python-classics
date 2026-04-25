import streamlit as st
import pandas as pd

st.set_page_config(page_title="IT Online Streamlit App 2", layout="wide")
st.title("Hello IT online Apps with Core Components Widget")

with st.sidebar:
    st.header("Settings")
    theme=st.selectbox("Theme", ["Light", "Dark", "Auto"], index=0)
    show_raw=st.checkbox("Showing data", value=True)

col1,col2=st.columns(2)
with col1:
    name=st.text_input("type your name", placeholder="Enter Name")

with col2:
    age=st.slider("Select your age", min_value=1, max_value=60, value=34)

lang=st.selectbox("Language selected", ["Python", "Java", "Go", "Typescript"])

exper_level=st.radio("Experience level", ["Beginner", "Intermediate", "Advance"], index=1)

#form submission
with st.form("Survey"):
    st.write("Quick Survey")
    enjoy_ds=st.checkbox("Enjoy working with data folks", value=True)
    hours=st.number_input("Hours/week learning", min_value=0, max_value=40)
    submitted=st.form_submit_button("Submit")

if submitted:
    st.success("Thanks for your informations")
    st.write({
        "name": name,
        "age": age,
        "language": lang,
        "experience_level": exper_level,
        "Enjoyed working with data folks": enjoy_ds,
        "Hours": hours,
        "theme": theme,
    })

df=pd.DataFrame({
    "city": ["Paris", "London", "Berlin", "Sweden"],
    "temp": [21, 24, 45,23],
    "raining": [False, False, True, False]
})
st.subheader("Data captured from London Dataset")
st.table(df if not show_raw else df.sort_values("temp", ascending=False))