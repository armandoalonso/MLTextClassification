import streamlit as st
import pandas as pd
import functions as f
from datetime import date


st.set_page_config(layout="wide")
st.title('Train Model')

if f.key_exists('data'):
    data = f.get_key('data')

    training_algorithm = st.selectbox('Training Algorithm', ['Multinomial Naive Bayes', 'Support Vector Classification', 'Logistic Regression'], key='Training Algorithm')

    with st.form(key='train_model_form'):
        col1, col2 = st.columns(2)

        with col1:
            features = st.selectbox('Features', data.columns, key='Features', index=0)
        with col2:
            target = st.selectbox('Target', data.columns, key='Target', index=1)

        test_size = st.slider('Test/Train Split', min_value=0.0, max_value=1.0, value=0.2, step=0.1, key='Test Size')
        training_opts = {}

        if training_algorithm == 'Multinomial Naive Bayes':
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                training_opts['ngram_start'] = st.number_input('ngram_start', value=1)
                training_opts['ngram_end'] = st.number_input('ngram_end', value=1)
            with col2:
                training_opts['k_features'] = st.selectbox('k_features', ['all', 'best'], key='k_features', index=1)
                training_opts['k_best'] = st.number_input('k (best features)', value=10)
            with col3:
                training_opts['alpha'] = st.number_input('alpha', value=1.0)
                training_opts['force_alpha'] = st.selectbox('force_alpha', ['True', 'False'], key='force_alpha', index=1)
            with col4:
                training_opts['fit_prior'] = st.selectbox('fit_prior', ['True', 'False'], key='fit_prior', index=1)

        if training_algorithm == 'Support Vector Classification':
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                training_opts['ngram_start'] = st.number_input('ngram_start', value=1)
                training_opts['ngram_end'] = st.number_input('ngram_end', value=1)
            with col2:
                training_opts['k_features'] = st.selectbox('k_features', ['all', 'best'], key='k_features', index=1)
                training_opts['k_best'] = st.number_input('k (best features)', value=10)
            with col3:
                training_opts['penalty'] = st.selectbox('penalty', ['l1', 'l2'], key='penalty', index=1)
                training_opts['loss'] = st.selectbox('loss', ['hinge', 'squared_hinge'], key='loss', index=1)
                training_opts['c'] = st.number_input('c', value=1.0)
            with col4:
                training_opts['max_iter'] = st.slider('max_iter', min_value=100, max_value=3000, value=1000, step=1)


        if training_algorithm == 'Logistic Regression':
            #st.markdown('The choice of the algorithm depends on the penalty chosen. Supported penalties by solver')
            #st.dataframe(pd.DataFrame({'lbfgs', 'l2, None', 'liblinear', ['l1', 'l2'], 'newton-cg', ['l2', 'None'], 'newton-cholesky', ['l2', 'None'], 'sag', ['l2', 'None'], 'saga', ['elasticnet', 'l1', 'l2', 'None']}))

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                training_opts['ngram_start'] = st.number_input('ngram_start', value=1)
                training_opts['ngram_end'] = st.number_input('ngram_end', value=1)
            with col2:
                training_opts['k_features'] = st.selectbox('k_features', ['all', 'best'], key='k_features', index=1)
                training_opts['k_best'] = st.number_input('k (best features)', value=10)
            with col3:
                training_opts['penalty'] = st.selectbox('penalty', ['none', 'l1', 'l2', 'elasticnet'], key='penalty', index=1)
                training_opts['solver'] = st.selectbox('solver', ['liblinear', 'saga'], key='solver', index=1)
                training_opts['c'] = st.number_input('c', value=1.0)
            with col4:
                training_opts['max_iter'] = st.slider('max_iter', min_value=100, max_value=3000, value=1000, step=1)
                

        train_model = st.form_submit_button(label='Train Model')
        if train_model:
            with st.spinner('Training Model...'):
                # split data
                X_train, X_test, y_train, y_test = f.split_data(data, features, target, test_size)

                # create pipeline
                pipline = f.create_pipeline(training_algorithm, training_opts)

                # train model
                model = f.train_model(pipline, X_train, y_train)

                # test model
                predicted = f.predict_set(model, X_test)

                # display predicted results
                st.subheader('Predicted')
                st.dataframe (pd.DataFrame({'Actual': y_test, 'Predicted': predicted}), use_container_width=True)

                # display model metrics
                st.subheader('Model Metrics')
                accuracy, precision, recall, f1 = f.get_model_metrics(y_test, predicted)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric('Accuracy: ', f.get_percent(accuracy))
                with col2:
                    st.metric('Precision: ', f.get_rounded_float(precision))
                with col3:
                    st.metric('Recall: ', f.get_rounded_float(recall))
                with col4:
                    st.metric('F1: ', f.get_rounded_float(f1))

                model_pickle = f.pickle_model(model)
                f.set_key('model', model_pickle)
                f.set_key('model_name', f.generate_model_name(training_algorithm))
                f.set_key('train_model', True)

    # download model button
    if f.get_key('train_model'):
        st.divider()
        model_pickle = f.get_key('model')
        filename = st.text_input('Filename', value=f.get_key('model_name'), key='filename')
        st.download_button(label='Download Model', data=model_pickle, file_name=filename, mime='application/octet-stream')


