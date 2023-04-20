import streamlit as st
import pandas as pd
import functions as f

st.set_page_config(layout="wide")
st.title('Load Data')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.dataframe(data)
    f.set_key('data', data)