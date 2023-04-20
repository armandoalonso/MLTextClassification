import streamlit as st
import pandas as pd
import functions as f
from datetime import date

st.set_page_config(layout="wide")
st.title('Test Classification Model')

if f.key_exists('model'):
    model = f.get_key('model')
    with st.form(key='test_model'):
        text = st.text_area('Enter Text', height=500)
        submit = st.form_submit_button(label='Predict')

        if submit:
            with st.spinner('Predicting...'):
                predicted = f.predict_text(model, text)
                st.success('Predicted Label: {}'.format(predicted))

else:
    st.subheader('Upload Model to Test')
    file = st.file_uploader('Upload Model', type=['pkl'])
    if file:
        model = f.unpickle_model(file)
        f.set_key('model', model)
