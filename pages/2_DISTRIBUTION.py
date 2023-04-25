import streamlit as st
import pandas as pd
import functions as f
from datetime import date

st.set_page_config(layout="wide")
st.title('Target Distribution')

if f.key_exists('data'):
    data = f.get_key('data')
    target_distribution = f.get_column_values(data, 'Target')
    max_target = f.get_max_column_value(data, 'Target')
    st.dataframe(target_distribution, use_container_width=True)

    with st.form(key='target_distribution_form'):
        col1, col2 = st.columns(2)

        with col1:
            lower_case_target = st.checkbox('Lower Case Target', key='lower_case_target')
            remove_punctuation = st.checkbox('Remove Punctuation', key='remove_punctuation')
            normalize_spaces = st.checkbox('Normalize Spaces', key='normalize_spaces')
            shuffle_data = st.checkbox('Shuffle Data', key='shuffle_data')
            balance_data = st.checkbox('Balance Data', key='balance_data')
        with col2:
            max_values_per_target = st.slider('Max Values Per Target', min_value=1, max_value=max_target, value=max_target, step=1)
        exclude_targets = st.multiselect('Exclude Targets', data["Target"].unique(), key='exclude_targets')

        target_distribution = st.form_submit_button(label='Normalize Target Distribution')
        if target_distribution:
            with st.spinner('Target Distribution...'):

                opts = {
                    'lower_case': lower_case_target,
                    'remove_punctuation': remove_punctuation,
                    'normalize_spaces': normalize_spaces,
                    'shuffle': shuffle_data,
                    'max_values_per_target': max_values_per_target,
                    'balance_data': balance_data,
                    'exclude_targets': exclude_targets
                }

                data = f.normalize_target_distribution(data, **opts)
                f.set_key('data', data)

                st.success('Target Distribution Normalized!')
                target_distribution = f.get_column_values(data, 'Target')
                st.dataframe(target_distribution, use_container_width=True)
                f.set_key('target_distribution', True)

    # download data button
    if f.get_key('target_distribution'):
        st.divider()
        data = f.get_key('data')
        filename = st.text_input('Filename', value='target-distribution-'+str(date.today())+'.csv')
        st.download_button(label='Download Data', data=data.to_csv(index=False), file_name=filename, mime='text/csv')