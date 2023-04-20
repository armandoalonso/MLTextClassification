import streamlit as st
import pandas as pd
import functions as f
from datetime import date

st.set_page_config(layout="wide")
st.title('Organize Data')

if f.key_exists('data'):
    data = f.get_key('data')
    
    with st.form(key='organize_data_form'):
        col1, col2 = st.columns(2)

        with col1:
            features = st.multiselect('Features', data.columns, key='Features')
        with col2:
            targets = st.multiselect('Target', data.columns, key='Target')

        organize_data = st.form_submit_button(label='Organize Data')
        if organize_data:
            with st.spinner('Organizing Data...'):
                # combine target columns
                data = f.fill_column_if_null(data, targets)
                data = f.combine_columns(data, targets, 'Target', '-')

                #combine features columns
                data = f.fill_column_if_null(data, features)
                data = f.combine_columns(data, features, 'Features', ', ')

                # drop unsed columns
                columns = ['Features', 'Target']
                data = data[columns]

                st.success('Data Organized!')
                st.dataframe(data, use_container_width=True)

                f.set_key('data', data)
                f.set_key('organize_data', True)

    # download data button
    if f.get_key('organize_data'):
        st.divider()
        data = f.get_key('data')
        filename = st.text_input('Filename', value='organized-data-'+str(date.today())+'.csv')
        st.download_button(label='Download Data', data=data.to_csv(index=False), file_name=filename, mime='text/csv')
    