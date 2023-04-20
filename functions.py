import io
import pickle
import string
import streamlit as st
import pandas as pd
import spacy
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
 
def tokenizer(sentence):
    punctuations = string.punctuation
    stop_words = spacy.lang.en.stop_words.STOP_WORDS
    parser = get_spacy_en_parser()
    tokens = parser(sentence)
    result = []
    for token in tokens:
        #check if token is a stop word or punctuation
        if token.text in stop_words or token.text in punctuations:
            continue
        #print(token.text)
        text = clean_token(token)
        if text == '':
            continue
        #print(text)
        result.append(token.lower_)
    return result
    
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        for text in X:
            ct = clean_text(text)
            yield ct
    def fit(self, X, y=None, **fit_params):
        return self
    def get_params(self, deep=True):
        return {}

def clean_text(text):
    #replace new lines with spaces
    text = text.replace('\r', ' ')
    text = text.replace('\n', ' ')
    #replace emails
    text = text.replace('\S*@\S*\s?', '')
    #remove punctuations
    text = text.replace('[^\w\s]',' ')
    #remove double spaces
    text = text.replace('\s+', ' ')
    #remove numbers
    text = text.replace('\d+', '')
    #remove single characters
    text = text.replace(r'\b[a-zA-Z]\b', '')
    #remove special characters
    text = text.replace('[^a-zA-Z]', '')
    
    return text.strip().lower()

def clean_token(token):
    text = token.text.replace('\r', ' ')
    text = text.replace('\n', ' ')
    text = text.replace('[^\w\s]',' ')
    text = text.replace('\s+', '')
    return token.text.strip().lower()


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
        return None

def create_svc_pipeline(training_opts):
    ngram_tuple = (training_opts['ngram_start'], training_opts['ngram_end'])
    k = training_opts['k']
    max_iter = training_opts['max_iter']
    c = training_opts['c']
    penalty = training_opts['penalty']
    loss = training_opts['loss']

    pipeline = Pipeline([('cleaner', predictors()),
                         ('vect', TfidfVectorizer(tokenizer=tokenizer, ngram_range=ngram_tuple, sublinear_tf=True)),
                         ('chi',  SelectKBest(chi2, k=k)),
                         ('clf', LinearSVC(C=c, penalty=penalty, max_iter=max_iter, loss=loss, dual=False))])
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