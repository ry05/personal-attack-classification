"""
Preprocessing script
--------------------

This script contains elements that help in preprocessing the text
"""

import re
import string

from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from afinn import Afinn
afinn = Afinn()

DEF_STOPWORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', "don't", 'should', "should've", 'now', "aren't", "couldn't", "didn't", "doesn't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't", "needn't", "shan't", "shouldn't", "wasn't", "weren't", "won't", "wouldn't"]

class StemTokenizer(object):
    """
    Stems tokens
    """

    def __init__(self):
        self.stemmer = SnowballStemmer(language='english')

    def __call__(self, comment):
        return [self.stemmer.stem(token) for token in word_tokenize(comment)]

class LemmaTokenizer(object):
    """Lemmatizes tokens
    Source: https://stackoverflow.com/questions/47423854/sklearn-adding-lemmatizer-to-countvectorizer
    """

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, comment):
        return [self.lemmatizer.lemmatize(token) for token in word_tokenize(comment)]


def remove_punct(text):
    """
    Removes punctuations from text
    :param text: text (str)
    :return: text with no punctuations
    """

    return  re.sub(f'[{re.escape(string.punctuation)}]', '', text)

def bad_spaces(text):
    """
    Remove bad spaces
    :param text: text (str)
    :return: text with no bad spacing
    """

    tokens = text.split()
    text = " ".join(tokens)
    return text

def make_lower(text):
    """
    Convert to lowercase
    :param text: text (str)
    :return: text in lowercase
    """

    return text.lower()

def translate_text(text):
    """
    Translate the text by performing "special operations"
    :param text: text (str)
    :return: translated text
    """

    # corpus stopwords obtained from data exploration
    corpus_stopwords = ['fuck', 'fag', 'faggot', 'fggt', 'nigga', 'nigger', 'aids', 'article', 'page', 'wiki', 'wp', 'block', 'NOES', 'ANONYMOUS', 'UTC', 'NOT', 'OH', 'IP', 'POV', 'LIVE', 'WP', 'REDIRECT', 'BTW', 'AIDS', 'HUGE', 'BLEACHANHERO', 'PHILIPPINESLONG']
    cs_lower = [s.lower() for s in corpus_stopwords]
    cs_upper = [s.upper() for s in corpus_stopwords]

    you_tokens = ['you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves']
    stop_tokens = DEF_STOPWORDS
    
    # remove punctuations
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)

    # remove corpus stopwords
    # removing these won't affect as the presence of necessary words have been computed in data exploration
    # and the dataset is stored
    text_tokens = text.split()
    text_tokens = [tok for tok in text_tokens if ((tok not in cs_lower) and (tok not in cs_upper))]
    translated_tokens = []

    # add labels to select groups of words
    for token in text_tokens:
        if token in you_tokens:
            translated_tokens.append("YOUWORD")
        elif token in stop_tokens:
            translated_tokens.append("STOPWORD")
        else:
            translated_tokens.append(token)

    translated_text = " ".join(translated_tokens)

    return translated_text
