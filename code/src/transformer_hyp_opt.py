""" 
Hyperparameter optimization for the Robertabase
transformer model

This script interfaces with the Weights and Biases
ML Experiment platform to track the performance of different
hyperparameter combinations

The final output of the performances are stored in
../model/hyp_opt_transformer.csv

The colab file used is at
https://colab.research.google.com/drive/1isI2ZvCre-J-1PXk-EGI4vdYyNd1fcrr?usp=sharing
"""

import pandas as pd
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.model_selection import StratifiedKFold
import wandb

def run_training():
    """
    Initialize a new weights&biases run
    """

    # initialize the project
    wandb.init()

    # create the robertabase model
    model = ClassificationModel(
        "roberta",
        "roberta-base",
        use_cuda=True,
        args=model_args,
        sweep_config=wandb.config,
    )

    # model training
    model.train_model(train, eval_df=val)

    # model validation
    model.eval_model(val)

    wandb.join()

if __name__ == '__main__':

    # get data
    df = pd.read_csv("../data/train.csv")

    # stratified cross validation
    df['kfold'] = -1

    target = df.attack.values
    kf = StratifiedKFold(n_splits=5)
    for fold, (trn_, val_) in enumerate(kf.split(X=df, y=target)):
        df.loc[val_, 'kfold'] = fold 

    df = df.drop(['id'], axis=1)
    df.columns = ['labels', 'text', 'kfold']

    # prep data
    train = df[df.kfold != 0]
    val = df[df.kfold == 0]

    # code for hyperparameter tuning
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "train_loss", "goal": "minimize"},
        "parameters": {
            "num_train_epochs": {"values": [1,2,3,5]},
            "learning_rate": {"min": 0, "max": 4e-5},
        }
    }
    # initialize sweep
    # a sweep is a hyperparameter optimization process in weights&biases terminology
    sweep_id = wandb.sweep(sweep_config, project="wiki_personal_attack")

    model_args = ClassificationArgs()
    # assign weights to the labels
    # higher weight to the positive class of being a personal attack
    model_args.weight = [0.5, 1]
    # set seed for reproducibility
    model_args.manual_seed = 42

    # begin the optimization
    wandb.agent(sweep_id, run_training)




