"""
This file contains code to create the deep learning
baseline model that scored 0.73 on the public leaderboard

This code was run in a kaggle kernel to make use of GPUs

Therefore, running this code on a local system might throw issues
with paths of data

If to_csv() throws an error, install pandas==1.1.5 just before
using to_csv()

NOTE: Most of the code in this file has been written with the help of
"Deep Learning for Coders with fastai & PyTorch" by Jeremy Howard and
Sylvian Gugger (2020)
"""

from fastai.text.all import *
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('../input/my474-classification-challenge-2021/train.csv')
df = df.drop(['id'], axis=1)

# separate out holdout set
train, valid = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df['attack']
)

print(f' The shape of training set is {train.shape}')
print(f' The shape of test set is {valid.shape}')

# create the 'is_valid' column and make the dataset whole again
# this operation is necessary to use fastai's API
train['is_valid'] = False
valid['is_valid'] = True
df = pd.concat([train, valid], axis=0)
df = df.sample(frac=1)

# Creating a Language Model

wiki_talks_lm = DataBlock(
    blocks=(TextBlock.from_df('text', is_lm=True)),
    get_x=ColReader('text'),
    splitter=ColSplitter()
).dataloaders(df, bs=128, seq_len=100)

# fine tune the language model
learn = language_model_learner(
    wiki_talks_lm, AWD_LSTM, drop_mult=0.5,
    metrics=[accuracy, Perplexity()]).to_fp16()

# fit one cycle with 11 epochs
learn.fit_one_cycle(11, 2e-2)

# save the encoder
learn.save_encoder('finetuned')

# Creating the Classifier object

# create classifier DataLoaders
wiki_talks_clas = DataBlock(
    blocks=(TextBlock.from_df('text', seq_len=100, vocab=wiki_talks_lm.vocab),CategoryBlock),
    get_x=ColReader('text'),
    get_y=ColReader('attack'),
    splitter=ColSplitter()
).dataloaders(df, bs=128)

learn = text_classifier_learner(wiki_talks_clas, AWD_LSTM, drop_mult=0.5, metrics=[accuracy, F1Score()]).to_fp16()
learn = learn.load_encoder('finetuned')

# fine tune the classifier
learn.fit_one_cycle(11, 2e-2)

# unfreeze model one by one to improve performance
learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-2/(2.6**4), 1e-2))
learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(5e-3/(2.6**4), 5e-3))
learn.freeze_to(-4)
learn.fit_one_cycle(1, slice(5e-3/(2.6**4), 5e-3))
learn.freeze_to(-5)
learn.fit_one_cycle(1, slice(5e-3/(2.6**4), 5e-3))
learn.freeze_to(-6)
learn.fit_one_cycle(1, slice(5e-3/(2.6**4), 5e-3))
learn.unfreeze()
learn.fit_one_cycle(2, slice(1e-3/(2.6**4), 1e-3))

# Using the Trained Model for Inference

test = pd.read_csv('../input/my474-classification-challenge-2021/test.csv')
submission = pd.read_csv('../input/my474-classification-challenge-2021/sampleSubmission.csv')

# make predictions
test_data = learn.dls.test_dl(test['text'])
preds = learn.get_preds(dl=test_data)
submission.to_csv('submission.csv', index=False)