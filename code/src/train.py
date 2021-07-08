"""
Training script

TODO
This is hardcoded as of now. In practice, it should access model_dispatcher.py or something
similar to extract the necessary model and preprocessing to get results.

Best to execute this script from a call in ../notebooks/modelling.ipynb by passing in
arguments like
1. training data 
2. preprocessing steps pipeline
3. model to use

A pipeline should be formed with these.

Display CV F1
"""

import os
import joblib
from collections import defaultdict

import config

import pandas as pd 
import numpy as np
from sklearn import metrics
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
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

class LemmaTokenizer(object):
    """Lemmatizes tokens
    Source: https://stackoverflow.com/questions/47423854/sklearn-adding-lemmatizer-to-countvectorizer
    """

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, comment):
        return [self.lemmatizer.lemmatize(token) for token in word_tokenize(comment)]

def run(fold):
    """
    Compute cross validation score for a fold
    :param fold: fold number (int)
    :param target: name of target feature (str)
    :return cv_score: cross validation score (float)
    """

    df = pd.read_csv(os.path.join("../data", config.FOLDS_DATA_PATH))
    target = config.TARGET
    clean = config.CLEANED_TEXT

    # override the above initializations with pre-cleaned data
    df = pd.read_csv("../data/train_folds.csv")
    target = 'attack'
    clean = 'text'

    # get train and validation
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_validation = df[df.kfold == fold].reset_index(drop=True)

    # preprocess text
    """
    df_train['clean'] = df_train.text.apply(preprocess)
    df_validation['clean'] = df_validation.text.apply(preprocess)
    """

    # heuristical measures; required only for test or validation sets
    df_validation['slur'] = df_validation[clean].apply(slur_exists)
    df_validation['profanity'] = df_validation[clean].apply(is_profanity)

    # feature representation (call it vector representation)
    feat_rep = TfidfVectorizer(
        tokenizer = StemTokenizer(),
        token_pattern = None,
        ngram_range=(1,1)
    )
    feat_rep.fit(df_train[clean])
    """
    feat_rep = TfidfVectorizer(
        tokenizer = StemTokenizer(),
        token_pattern = None
    )
    feat_rep.fit(df_train[clean])
    """

    # split into train and validation
    train_x = feat_rep.transform(df_train[clean])
    train_y = df_train[target].values
    validation_x = feat_rep.transform(df_validation[clean])
    validation_y = df_validation[target].values

    print("With no feature selection")
    print(train_x.shape)
    print(validation_x.shape)

    """
    # dimensionality reduction
    svd = TruncatedSVD(n_components=100, n_iter=5, random_state=42)
    svd.fit(train_x)
    train_x = svd.transform(train_x)
    validation_x = svd.transform(validation_x)
    """

    # feature selection

    # univariate feature selection
    ufs = UFS(
        n_features = 0.05, # top 1% is good as it reduces the feature space well and also performs similar to having top 10% features
        scoring = 'chi2'
    )
    ufs.fit(train_x, train_y)
    train_x = ufs.transform(train_x)
    validation_x = ufs.transform(validation_x)

    print("Feature selection with chi2 univariate selection - Top 1% features")
    print(train_x.shape)
    print(validation_x.shape)

    
    # l1-based feature selection
    lsvc = LinearSVC(C=0.1, penalty="l2", dual=False).fit(train_x, train_y)
    model = SelectFromModel(lsvc, prefit=True)
    train_x = model.transform(train_x)
    validation_x = model.transform(validation_x)

    print("After l1-based LinearSVC feature selection")
    print(train_x.shape)
    print(validation_x.shape)

    # hyperparameter optimization
    # done in ../data/modelling.ipynb

    # classifier
    clf = ComplementNB(alpha=0.01)
    # fit
    clf.fit(train_x, train_y)

    # 
    # predictions
    preds = clf.predict(validation_x)

    # heuristicals
    slur = df_validation['slur']
    profanity = df_validation['profanity']
    #article = df_validation['article']

    # potentially_problematic words

    # update preds
    preds = preds | slur # if model has predicted 1, use it. Else, if slur or profane word exists, its attack
    
    # find a way to store model

    # compute metric (f1)
    f1 = metrics.f1_score(validation_y, preds)
    acc = metrics.accuracy_score(validation_y, preds)

    # print the confusion matrix
    print(f'Confusion matrix for fold {fold}')
    print(metrics.confusion_matrix(validation_y, preds))
    
    return f1, acc

def get_cv_score():
    """
    Get cross validation scores
    :return cv_scores: cross validation scores (list)
    """

    cv_f1_scores = defaultdict(list)
    cv_acc_scores = defaultdict(list)
    for fold in range(config.NFOLDS):
        f1, acc = run(fold)
        cv_f1_scores[fold+1].append(f1)
        cv_acc_scores[fold+1].append(acc)

    #return cv_scores

    print(f'The list of F1 CV scores for 5-fold CV is {cv_f1_scores}')
    print(f'The mean F1 CV score is {np.mean(list(cv_f1_scores.values()))}')
    print(f'The list of CV accuracies for 5-fold CV is {cv_acc_scores}')
    print(f'The mean CV accuracy is {np.mean(list(cv_acc_scores.values()))}')

if __name__ == '__main__':

    get_cv_score()