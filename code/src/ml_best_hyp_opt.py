"""
Hyperparameter Tuning of the Best Model
"""

import pandas as pd
from fastai.text.all import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import  ComplementNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler # to get between 0 and 1; afinn can take negative values

import config

# fastai's tokenizer function
spacy = WordTokenizer()
tokenizer = Tokenizer(spacy)

if __name__ == "__main__":

    # data preparation
    train = pd.read_csv(config.TRAIN_NUMERIC)
    train = train.fillna('missing')
    target = config.TARGET
    text_feat = config.TEXT
    train_x = train.drop(['id', 'attack'], axis=1)
    train_y = train[target].values

    # get the best pipeline
    numeric_transformer = Pipeline([
        ('scaler', MinMaxScaler())
    ])

    text_transformer = Pipeline([
        ('vect', CountVectorizer(
            tokenizer=tokenizer,
            max_features=10000
        )),
        ('tfidf', TfidfTransformer()),
        ('ufs', SelectPercentile(
            score_func=chi2
        ))
    ])

    # preprocessor
    preprocessor = ColumnTransformer(
        transformers = [
            ('num', numeric_transformer, numeric),
            ('text', text_transformer, text)
    ])

    # classifier
    clf = Pipeline([
        ('preprocess', preprocessor),
        ('clf', ComplementNB()) # ComplementNB was used
    ])

    numeric = ['afinn', 'you_count', 'caps_word_count', 'digits_count', 'dale_chall']
    binary = ['source_cnt', 'f*g_cnt', 'n***_cnt', 'fu**_cnt', 'article_cnt', 'REDIRECT_count'] # these are not preprocessed
    text = 'text'
    # preprocessor for heterogenous data
    preprocessor = ColumnTransformer(
        transformers = [
            ('num', numeric_transformer, numeric),
            ('text', text_transformer, text)
    ])
    # integrate it all into the pipeline
    estimator = Pipeline([
        ('preprocess', preprocessor),
        ('clf', ComplementNB(alpha=0.05)) # ComplementNB was used
    ])

    # hyperparameter space
    params = {
        'preprocess__text__ufs__percentile': [1, 2, 5, 10, 15, 20, 50],
        'clf__alpha': (2, 1.5, 1, 1e-1, 3e-1, 5e-1, 1e-2, 3e-2, 5e-2, 1e-3, 3e-3, 5e-3, 0)
    }
    # grid search cv
    rs_clf = GridSearchCV(clf, params, cv=5, n_jobs=-1, scoring='f1', error_score='raise')
    rs_clf = rs_clf.fit(X, y)

    # convert CV results to dataframe and store
    cv_res = pd.DataFrame(rs_clf.cv_results_)
    cv_res = cv_res.dropna()
    cv_res.to_csv('../models/hyp_opt_ml.csv', index=False)