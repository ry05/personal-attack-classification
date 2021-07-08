#!/bin/sh

pip install fastai
python -m nltk.downloader all
pip install -U spacy
python -m spacy download en_core_web_md
pip install wordcloud
pip install textstat
pip install texthero
pip install "gensim==3.8.1"
pip install argparse
pip install unidecode
pip install afinn
pip install textdistance
pip install SciencePlots
pip install statsmodels