"""
Inference script
"""

import argparse

import pandas as pd 
from sklearn.model_selection import cross_val_score

import config
import pipelines as pipe
from preprocess import translate_text

if __name__ == "__main__":

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
        test = pd.read_csv(config.TEST_RAW)
        train = train.fillna('missing')
        test = test.fillna('missing')
        # split into independent and dependent variables
        train_x = train[text_feat]
        train_y = train[target].values
        test_x = test[text_feat]
    elif args.data == 'with_numeric':
        train = pd.read_csv(config.TRAIN_NUMERIC)
        test = pd.read_csv(config.TEST_NUMERIC)
        train = train.fillna('missing')
        test = test.fillna('missing')
        # split into independent and dependent variables
        train_x = train.drop(['id'], axis=1)
        train_y = train[target].values
        test_x = test[text_feat]
    elif args.data == 'with_numeric_translated_text':
        train = pd.read_csv(config.TRAIN_NUMERIC)
        test = pd.read_csv(config.TEST_NUMERIC)
        train = train.fillna('missing')
        test = test.fillna('missing')
        # split into independent and dependent variables
        train[text_feat] = train[text_feat].apply(translate_text)
        test[text_feat] = test[text_feat].apply(translate_text)
        train_x = train.drop(['id'], axis=1)
        train_y = train[target].values 
        test_x = train.drop(['id'], axis=1)  

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

    # fit estimator on training data
    print("Fitting the estimator on the training data...")
    estimator.fit(train_x, train_y)

    print("Fitting complete.")
    print(train_x.shape)
    print(test_x.shape)

    # get predictions
    print("Predicting for the test data...")
    preds = estimator.predict(test_x)

    # make submission file
    ids = test['id'].values
    sub = pd.DataFrame({
        'id': ids,
        'attack': preds
    })
    sub.to_csv('../data/submission.csv', index=False)
    print("Prediction complete. Submission file generated in the data folder.")
    print(f"The submission has {sub.shape[0]} predictions")