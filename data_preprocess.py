import numpy as np
import pandas as pd

from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
from nltk.stem.snowball import FrenchStemmer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict

import os
import nltk
import string
import nltk
import math
import re


# convert text to lower case
def convert_lower_case(data):
    return np.char.lower(data)


def remove_french_stop_words(data):
    stop_words = stopwords.words('french')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1 and not w.isnumeric():
            new_text = new_text + " " + w
    return new_text


# remove punctuation

def remove_punctuation(data):
    text = word_tokenize(str(data))
    new_words = []
    for word in text:
        w = re.sub(r'\/', ' ', word)
        w = re.sub(r'[^\w\s]', '', w)  # remove everything except words and space
        w = re.sub(r'\_', '', w)  # to remove underscore as well
        new_words.append(w)
    new_words = " ".join(new_words)

    return new_words


def remove_extra_whitespace_tabs(data):
    pattern = r'^\s*|\s\s*'
    return re.sub(pattern, ' ', data).strip()


# remove apostrophe for french (l'..., qu'..., d'...)
def remove_apostrophe(data):
    return np.char.replace(data, "'", " ")


# perform stemming using porterstemmer
def french_stemming(data):
    stemmer = FrenchStemmer('french')
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text


def preprocess(data, language):
    if language == 'fr':
        data = convert_lower_case(data)
        data = remove_apostrophe(data)
        data = remove_punctuation(data)
        data = remove_french_stop_words(data)
        data = french_stemming(data)
        data = remove_extra_whitespace_tabs(data)

        return data


def return_preprocessed_text(content, language):
    preprocessed_content = []

    for i in range(len(content)):
        text = content[i]
        preprocessed_content.append(preprocess(text, language))

    return preprocessed_content

