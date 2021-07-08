"""
This file contains code to create the transformers
model that scored 0.782 on the public leaderboard

The model uses a RoBERTa base pretrained architecture with
a  RoBERTa model

This code was run in Google Colab to make use of GPUs

Therefore, running this code on a local system might throw issues
with paths of data

NOTE: Code has been written on the basis of documentation from
https://simpletransformers.ai/

The hyperparameter optimization process to decide the right values for
learning rate and number of epochs is depicted in the transformer_hyp_opt.py
file

The colab file used is at
https://colab.research.google.com/drive/1isI2ZvCre-J-1PXk-EGI4vdYyNd1fcrr?usp=sharing
"""

import pandas as pd
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.model_selection import train_test_split, StratifiedKFold
import sklearn

df = pd.read_csv('../data/train.csv')
test = pd.read_csv("../data/test.csv")
sub = pd.read_csv("../data/sampleSubmission.csv")

# stratified cross validation
df['kfold'] = -1

target = df.attack.values
kf = StratifiedKFold(n_splits=5)
for fold, (trn_, val_) in enumerate(kf.split(X=df, y=target)):
    df.loc[val_, 'kfold'] = fold 

df = df.drop(['id'], axis=1)
df.columns = ['labels', 'text', 'kfold']

# use fold 0 as the only fold for training
# so essentially, this works like a holdout cross validation technique
train = df[df.kfold != 0]
val = df[df.kfold == 0]

# model configuration
# hyperparams set after hyperparam optimization - check ../models/hyp_opt_transformer.csv
model_args = ClassificationArgs()
model_args = {
    "num_train_epochs": 2,
    "learning_rate": 0.000017,
}

# model
model = ClassificationModel(
    "roberta", "roberta-base", args=model_args
)

# train
model.train_model(train, f1=sklearn.metrics.f1_score)

# validate
result, model_outputs, wrong_predictions = model.eval_model(val, f1=sklearn.metrics.f1_score)

# making a submission
test_comments = list(test['text'].values)
predictions, raw_outputs = model.predict(test_comments)
sub['attack'] = predictions
sub.to_csv("transformer_submission.csv", index=False)