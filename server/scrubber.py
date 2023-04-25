import spacy
import re
import string
from spacy.lang.en import English

nlp = spacy.load('en_core_web_lg')
parser = English()

def text_to_lower_case(text):
    return text.lower()

def remove_urls_from_text(text):
    return re.sub(r'http\S+', '', text)

def remove_punctuation_from_text(text):
    result = []
    for char in text:
        if char not in string.punctuation:
            result.append(char)
    return "".join(result)

def remove_numbers_from_text(text):
    result = []
    for char in text:
        if char not in string.digits:
            result.append(char)
    return "".join(result)

def normalize_spaces_in_text(text):
    return re.sub('\s+', ' ', text).strip()

def remove_emails_from_text(text):
    return re.sub(r'[\w\.-]+@[\w\.-]+', '', text) 

def get_lemmatized_text(sentence):
    doc = nlp(sentence)
    results = []
    for token in doc:
        results.append(token.lemma_.strip())
    return " ".join(results)

def remove_stop_words(text):
    results = []
    doc = nlp(text)
    for token in doc:
        if token.text.strip() not in nlp.Defaults.stop_words:
            results.append(token.text.strip())
    return " ".join(results)

def remove_entities_tokens(text):
    doc = nlp(text)
    results = []
    for token in doc:
        if token.ent_type_ == "":
            results.append(token.text.strip())
    return " ".join(results)

def transform_data(text, actionList):
    for action in actionList:
        text = action(text)
    return text

def remove_cutsom_tokens(text, custom):
    doc = nlp(text)
    results = []
    for token in doc:
        if token.text.strip() not in custom:
            results.append(token.text.strip())
    return " ".join(results)

def get_unique_tokens(text):
    doc = nlp(text)
    results = []
    for token in doc:
        if token.text.strip() not in results:
            results.append(token.text.strip())
    return " ".join(results)

def scrub_text(text, opts):
    actionList = []
    custom = opts['remove_custom'].split(",")

    if opts['lower_case']:
        actionList.append(text_to_lower_case)
    if opts['remove_punctuation']:
        actionList.append(remove_punctuation_from_text)
    if opts['remove_numbers']:
        actionList.append(remove_numbers_from_text)
    if opts['normalize_spaces']:
        actionList.append(normalize_spaces_in_text)
    if opts['remove_emails']:
        actionList.append(remove_emails_from_text)
    if opts['remove_urls']:
        actionList.append(remove_urls_from_text)
    if opts['lemmatize']:
        actionList.append(get_lemmatized_text)
    if opts['remove_stop_words']:
        actionList.append(remove_stop_words)
    if opts['remove_entities_tokens']:
        actionList.append(remove_entities_tokens)
    if opts['unique_tokens']:
        actionList.append(get_unique_tokens)

    text = transform_data(text, actionList)
    text = remove_cutsom_tokens(text, custom)

    return text