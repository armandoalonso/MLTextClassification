import streamlit as st
import pandas as pd
import functions as f
from datetime import date

st.set_page_config(layout="wide")
st.title('Prepare Data')

if f.key_exists('data'):
    data = f.get_key('data')

    with st.form(key='perpare_data'):
        col1, col2, col3 = st.columns(3)

        features = st.multiselect('Features', data.columns, key='Features')

        with col1:
            lower_case = st.checkbox('Lower Case', value=True, key='Lower Case')
            remove_punctuation = st.checkbox('Remove Punctuation', value=True, key='Remove Punctuation')
            remove_numbers = st.checkbox('Remove Numbers', value=True, key='Remove Numbers')  
        with col2:
            lemma = st.checkbox('Lemmatize', value=True, key='Lemmatize')
            remove_stopwords = st.checkbox('Remove Stopwords', value=True, key='Remove Stopwords')
            remove_urls = st.checkbox('Remove URLs', value=True, key='Remove URLs')
            #remove_single_characters_token = st.checkbox('Remove Single Characters', value=True, key='Remove Single Characters')
        with col3:
            normalize_spaces = st.checkbox('Normalize Spaces', value=True, key='Normalize Spaces')
            remove_emails = st.checkbox('Remove Emails', value=True, key='Remove Emails')
            remove_entities_tokens = st.checkbox('Remove Entities', value=True, key='Remove Entities')

        remove_custom = st.text_input('Remove Custom Words', value='', key='Remove Custom')

        if st.form_submit_button(label='Prepare Data'):
            data = f.preprocess_text_df(data, {
                'lower_case': lower_case,  'remove_punctuation': remove_punctuation, 
                'remove_stop_words': remove_stopwords, 'remove_numbers': remove_numbers, 
                'normalize_spaces': normalize_spaces,  'remove_emails': remove_emails, 'remove_urls': remove_urls,
                'lemmatize': lemma,  'remove_entities_tokens': remove_entities_tokens, 'features': features,
                'custom': remove_custom
            })

            f.set_key('data', data)
            st.success('Data prepared')
            st.dataframe(data, use_container_width=True)

    if f.key_exists('data'):
        st.divider()
        data = f.get_key('data')
        filename = st.text_input('Filename', value='prepared-data-'+str(date.today())+'.csv')
        st.download_button(label='Download Data', data=data.to_csv(index=False), file_name=filename, mime='text/csv')