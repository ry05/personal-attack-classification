"""
Training script
"""

import sys
import argparse

import pandas as pd 
from sklearn.model_selection import cross_validate

import config
import pipelines as pipe
from preprocess import translate_text

if __name__ == "__main__":

    # take in arguments from the terminal
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline_number",
        type=str
    )
    parser.add_argument(
        "--data",
        type=str
    )
    args = parser.parse_args()

    target = config.TARGET
    text_feat = config.TEXT

    # get the data
    if args.data == 'raw':
        train = pd.read_csv(config.TRAIN_RAW)
        train = train.fillna('missing')
        print('Preview of data')
        print(train.head())
        print()
        # split into independent and dependent variables
        train_x = train[text_feat]
        train_y = train[target].values
    elif args.data == 'with_numeric':
        train = pd.read_csv(config.TRAIN_NUMERIC)
        train = train.fillna('missing')
        print('Preview of data')
        print(train.head())
        print()
        # split into independent and dependent variables
        train_x = train.drop(['id', 'attack'], axis=1)
        train_y = train[target].values
    elif args.data == 'with_numeric_translated_text':
        train = pd.read_csv(config.TRAIN_NUMERIC)
        train = train.fillna('missing')
        print('Preview of data')
        print(train.head())
        print()
        # split into independent and dependent variables
        train[text_feat] = train[text_feat].apply(translate_text)
        train_x = train.drop(['id', 'attack'], axis=1)
        train_y = train[target].values    

    # instantiate the estimator
    if int(args.pipeline_number) < 4:
        if args.data != 'raw':
            print('Thou shall not use this combination of pipeline and data!')
            sys.exit(0)
        else:
            estimator = pipe.pipe_dict[args.pipeline_number]
    elif int(args.pipeline_number) == 4:
        print("Going ahead with this hoping you have rechecked `pipelines.py` w.r.t features considered!")
        if args.data == 'raw':
            print('Thou shall not use this combination of pipeline and data!')
            sys.exit(0)
        else:
            estimator = pipe.pipe_dict[args.pipeline_number]
    else:
        print('The pipeline or data or both have not yet been created!')
        sys.exit(0)

    print(f'Training {args.data} data with pipeline {args.pipeline_number} using 5-fold stratified cross validation...')
    print()

    # setup 5-fold stratified cross validation
    # stratified due to the presence of class imbalance
    cv_scores = cross_validate(estimator, train_x, train_y, cv=5, scoring=['f1', 'accuracy'], error_score='raise')
    performance_df = pd.DataFrame(
        dict(
            fit_time = cv_scores['fit_time'],
            score_time = cv_scores['score_time'],
            validation_acc = cv_scores['test_accuracy'],
            validation_f1 = cv_scores['test_f1']
        )
    )
    print(f'Performance Table')
    print('(All times in seconds)')
    print()
    print(performance_df)
    print()
    print(f'The mean CV F1 score is {performance_df.validation_f1.mean()}')
    print(f'The mean CV accuracy for 5-fold CV is {performance_df.validation_acc.mean()}')
    