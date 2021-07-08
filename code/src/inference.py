""" 
This script is to be run from the terminal
`python inference.py`

TODO
1. Add arguments into the script that will help choose which model to use
2. A better display of results
"""

import config
import joblib
from collections import defaultdict

import pandas as pd 
from sklearn import metrics, tree
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split

from preprocess import preprocess
from heuristical import slur_exists

import config
import joblib
from collections import defaultdict

import pandas as pd 
from sklearn import metrics, tree
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split

from preprocess import preprocess
from heuristical import slur_exists

import os
import joblib
from collections import defaultdict

import config

import pandas as pd 
import numpy as np
from sklearn import metrics, tree
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords 

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import naive_bayes
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Perceptron
from sklearn.feature_selection import VarianceThreshold
from sklearn import model_selection

from preprocess import preprocess
from heuristical import slur_exists, is_profanity
from feature_selection import UnivariateFeatureSelection as UFS

STOPWORDS = set(stopwords.words('english'))

class StemTokenizer(object):
    """Stems tokens
    """

    def __init__(self):
        self.stemmer = SnowballStemmer(language='english')

    def __call__(self, comment):
        return [self.stemmer.stem(token) for token in word_tokenize(comment)]

if __name__ == '__main__':

    train = pd.read_csv(config.CLEANED_TRAINING_DATA)
    test = pd.read_csv(config.CLEANED_TEST_DATA)
    train = train.fillna('missing_data')
    test = test.fillna('missing_data')
    clean = config.CLEANED_TEXT
    target = config.TARGET

    # overriding
    train = pd.read_csv("../data/train.csv")
    test = pd.read_csv("../data/test.csv")
    clean = 'text'
    target = 'attack'
    
    # preprocess
    """
    train['clean'] = train.text.apply(preprocess)
    test['clean'] = test.text.apply(preprocess)
    """

    # heuristical
    test['slur'] = test[clean].apply(slur_exists)

    # feature representation (call it vector representation)
    feat_rep = TfidfVectorizer(
        tokenizer = StemTokenizer(),
        token_pattern = None,
        ngram_range=(1,1)
    )
    feat_rep.fit(train[clean]) 
    
    # transform
    train_x = feat_rep.transform(train[clean])
    train_y = train[target]
    test_x = feat_rep.transform(test[clean])

    # univariate feature selection
    ufs = UFS(
        n_features = 0.05, # top 1% is good as it reduces the feature space well and also performs similar to having top 10% features
        scoring = 'chi2'
    )
    ufs.fit(train_x, train_y)
    train_x = ufs.transform(train_x)
    test_x = ufs.transform(test_x)

    # l1-based feature selection
    lsvc = LinearSVC(C=0.1, penalty="l2", dual=False).fit(train_x, train_y)
    model = SelectFromModel(lsvc, prefit=True)
    train_x = model.transform(train_x)
    test_x = model.transform(test_x)

    # classifier
    clf = naive_bayes.ComplementNB(alpha=0.01)
    # fit
    clf.fit(train_x, train_y)

    # heuristics
    slur = test['slur']

    # predictions on test data
    preds = clf.predict(test_x)
    preds = preds | slur

    # make submission file
    ids = test['id'].values
    sub = pd.DataFrame({
        'id': ids,
        'attack': preds
    })
    sub.to_csv('../data/submission.csv', index=False)
    print("Prediction complete.")