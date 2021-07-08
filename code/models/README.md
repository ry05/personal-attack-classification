# Descriptions of models used

> NOTE: 'Model' in this context refers to the complete end-to-end pipeline including preprocessing.

## Contents of this folder

1. hyp_opt_ml.csv - Table of hyperparameter combinations and performances for the best ML model
2. hyp_opt_transformer.csv - Table of hyperparameter combinations and performances for the best DL model
3. my474_models.csv - Table of performances of different ML models tried out

## Machine Learning Model Performance Table

The complete performance table of the different machine learning models tried during the course of this competition is stored in the `my474_models.xlsx` file within this folder. Only those models that were submitted to the competition have F1 public leaderboard scores.

The best ML model uses
- Bag of words feature representation weighted by [Tf-idf](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- Tokenized with the [fast.ai tokenizer](https://docs.fast.ai/text.core.html)
- Includes numerical features
- [Feature selection](https://scikit-learn.org/stable/modules/feature_selection.html)
    - Univariate feature selection with chi2
    - Model-based selection with linearSVC and 'l2' regularization
- [Complement Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html#sklearn.naive_bayes.ComplementNB)

It scored 69.086% on the public leaderboard

## Deep Learning Models

1. [ULM-FiT](https://paperswithcode.com/method/ulmfit) => Best F1 on public leaderboard: 73%
    - Code in `../src/deeplearning_ulmfit.py`
2. [RoBERTa](https://paperswithcode.com/method/roberta) => Best F1 on public leaderboard: 78.2%
    - Code in `../src/deeplearning_transformer.py`

## Quick Notes about Modelling

### Performance 
- The best DL model implemented in this project is ~9% higher than the best ML model implemented in terms of F1 score
- The best ML model timed a mean training time of ~1600 seconds for 12000 samples and a mean inference time of ~400 seconds for 100000 samples
- The best DL model took ~2000 seconds to run inference on 100000 samples

### Complexity
- DL models are far more complex in terms of memory requirements as well as computational time requirements
- All the ML models could be run using a CPU while the DL models required a GPU (thus were run on Kaggle and Google Colab)

### Interpretability
- DL models are also much harder to interpret, making them less preferred than ML models if explainability is of importance


