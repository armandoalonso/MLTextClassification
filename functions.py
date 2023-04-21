import io
import pickle
import string
import streamlit as st
import pandas as pd
from datetime import date
import spacy
import re
from spacy.lang.en import English
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline

@st.cache_resource
def load_spacy_model():
    nlp = spacy.load('en_core_web_lg')
    return nlp

def get_spacy_en_parser():
    parser = English()
    return parser

def set_key(key, value):
    st.session_state[key] = value

def get_key(key):
    if key_exists(key):
        return st.session_state[key]
    else:
        return None

def key_exists(key):
    return key in st.session_state

def check_if_value_is_null(value):
    if pd.isnull(value) or value == ' ' or value == 'na' or value == 'N/A' or value == 'n/a' or value == 'NA' or value == 'NaN' or value == 'nan':
        return True
    else:
        return False

def fill_column_if_null(df, column_names, value=" "):
    for i in range(len(df)):
        for col in column_names:
            if check_if_value_is_null(df[col][i]):
                df[col][i] = value
    return df

def combine_columns(df, column_names, new_column_name, delimiter=' '):
    print('Combining columns: ' + str(column_names) + ' into ' + new_column_name)
    df[new_column_name] = df[column_names].agg(delimiter.join, axis=1)
    return df

def get_column_values(df, column_name):
    return df[column_name].value_counts()

def get_max_column_value(df, column_name):
    return int(df.groupby(column_name).size().max())

def normalize_spaces(df, column_name):
    df[column_name] = df[column_name].str.replace('\s+', ' ')
    return df

def lower_case_text(df, column_name):
    df[column_name] = df[column_name].str.lower()
    return df

def remove_punctuation(df, column_name):
    df[column_name] = df[column_name].str.replace('[^\w\s]',' ')
    return df

def remove_numbers(df, column_name):
    df[column_name] = df[column_name].str.replace('\d+', '')
    return df

def get_percent(float_value):
    return '{:.2%}'.format(float_value) 

def get_rounded_float(float_value):
    return '{:.2}'.format(float_value) 

def sampling_k_elements_from_list(group, k=50, balance=False):
    if len(group) < k:
        if balance:
            return group.sample(k, replace=True)
        else:
            return group
    else:
        return group.sample(k, replace=False)

def normalize_target_distribution(df, **kwargs):
    # define targe  column
    target = "Target"
    #shuffle dataframe
    if 'shuffle' in kwargs and kwargs['shuffle']:
        df = df.sample(frac=1).reset_index(drop=True)
 
    #iterate through each row
    for i in range(len(df)):
        #exclude targets
        if df[target][i] in kwargs['exclude_targets']:
            df = df.drop(i)
         
        if 'lower_case' in kwargs and kwargs['lower_case']:
            # lower case text
            df[target][i] = df[target][i].lower()
 
        if 'remove_punctuation' in kwargs and kwargs['remove_punctuation']:
            # remove punctuation
            df[target][i] = df[target][i].replace('[^\w\s]',' ')
         
        if 'normalize_spaces' in kwargs and kwargs['normalize_spaces']:
            # normalize spaces
            df[target][i] = df[target][i].replace('\s+', ' ')
  
    df = df.groupby(target).apply(sampling_k_elements_from_list, k=kwargs['max_values_per_target'], balance=kwargs['balance_data']).reset_index(drop=True)
    return df

def get_lemmatized_text(sentence):
    nlp = load_spacy_model()
    doc = nlp(sentence)
    results = []
    for token in doc:
        results.append(token.lemma_)
    return results

def to_lower_case(tokens):
    results = []
    for token in tokens:
        results.append(token.strip().lower())
    return results

def remove_stop_words(tokens):
    nlp = load_spacy_model()
    results = []
    for token in tokens:
        if token not in nlp.Defaults.stop_words:
            results.append(token)
    return results

def remove_single_characters(tokens):
    results = []
    for token in tokens:
        if len(token) > 1:
            results.append(token)
    return results

def remove_entities(tokens):
    nlp = load_spacy_model()
    doc = nlp(" ".join(tokens))
    results = []
    for token in doc:
        if token.ent_type_ == "":
            results.append(token.text)
    return results

def remove_ppi(tokens):
    results = []
    for token in tokens:
        # remove social security numbers
        if not re.match(r'^\d{3}-\d{2}-\d{4}$', token):
            # remove phone numbers
            if not re.match(r'^\d{3}-\d{3}-\d{4}$', token):
                # remove emails
                if not re.match(r'^\S+@\S+$', token):
                    results.append(token)
    return results

def remove_numbers(tokens):
    results = []
    for token in tokens:
        if not token.isnumeric():
            results.append(token)
    return results

def tokenizer(sentence):
    tokens = get_lemmatized_text(sentence)
    tokens = to_lower_case(tokens)
    tokens = remove_stop_words(tokens)
    tokens = remove_single_characters(tokens)
    tokens = remove_entities(tokens)
    tokens = remove_ppi(tokens) 
    tokens = remove_numbers(tokens)
    return tokens

def train_model(pipeline, X_train, y_train):
    return pipeline.fit(X_train, y_train)

def predict_set(model, X_test):
    return model.predict(X_test)

def predict_text(model, text):
    return model.predict([text])

def get_model_metrics(labels, predicted):
    accuracy = metrics.accuracy_score(labels, predicted)
    precision = metrics.precision_score(labels, predicted, average='weighted')
    recall = metrics.recall_score(labels, predicted, average='weighted')
    f1 = metrics.f1_score(labels, predicted, average='weighted')
    return accuracy, precision, recall, f1

def split_data(df, features, target, test_size=0.2):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

def create_pipeline(training_algorithm, training_opts):
    if training_algorithm == 'Support Vector Classification':
        return create_svc_pipeline(training_opts)
    elif training_algorithm == 'Multinomial Naive Bayes':
        return create_multi_nb_pipeline(training_opts)
    elif training_algorithm == 'Logistic Regression':
        return create_log_reg_pipeline(training_opts)

def create_svc_pipeline(training_opts):
    ngram_tuple = (training_opts['ngram_start'], training_opts['ngram_end'])
    k_features = training_opts['k_features']
    k = 'all' if k_features == 'all' else training_opts['k_best']
    max_iter = training_opts['max_iter']
    c = training_opts['c']
    penalty = training_opts['penalty']
    loss = training_opts['loss']

    pipeline = Pipeline([('vect', TfidfVectorizer(tokenizer=tokenizer, ngram_range=ngram_tuple, sublinear_tf=True)),
                         ('chi',  SelectKBest(chi2, k=k)),
                         ('clf', LinearSVC(C=c, penalty=penalty, max_iter=max_iter, loss=loss, dual=False))])
    return pipeline

def create_multi_nb_pipeline(training_opts):
    ngram_tuple = (training_opts['ngram_start'], training_opts['ngram_end'])
    k_features = training_opts['k_features']
    k = 'all' if k_features == 'all' else training_opts['k_best']
    alpha = training_opts['alpha']
    force_alpha = training_opts['force_alpha'] == 'True'
    fit_prior = training_opts['fit_prior'] == 'True'

    pipeline = Pipeline([('vect', TfidfVectorizer(tokenizer=tokenizer, ngram_range=ngram_tuple, sublinear_tf=True)),
                         ('chi',  SelectKBest(chi2, k=k)),
                         ('clf', MultinomialNB(alpha=alpha, force_alpha=force_alpha, fit_prior=fit_prior))])
    return pipeline

def create_log_reg_pipeline(training_opts):
    ngram_tuple = (training_opts['ngram_start'], training_opts['ngram_end'])
    k_features = training_opts['k_features']
    k = 'all' if k_features == 'all' else training_opts['k_best']
    max_iter = training_opts['max_iter']
    c = training_opts['c']
    penalty = training_opts['penalty']
    solver = training_opts['solver']

    pipeline = Pipeline([('vect', TfidfVectorizer(tokenizer=tokenizer, ngram_range=ngram_tuple, sublinear_tf=True)),
                         ('chi',  SelectKBest(chi2, k=k)),
                         ('clf', LogisticRegression(C=c, penalty=penalty, max_iter=max_iter, solver=solver))])
    return pipeline

def pickle_model(model):
    f = io.BytesIO()
    pickle.dump(model, f)
    return f

def unpickle_model(model_byte):
    model = pickle.load(model_byte)
    return model

def load_model(model_name):
    with open(model_name, 'rb') as file:
        model = pickle.load(file)
    return model

def generate_model_name(algorithm):
    #remove spaces
    algorithm = algorithm.replace(' ', '_').lower()
    return algorithm+'_'+str(date.today())+'.pkl'