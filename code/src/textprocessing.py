"""
Build the text processing pipeline

Remove this
"""
import argparse
import os

import config

import pandas as pd
import texthero as th 
from texthero import stopwords, preprocessing 

STOPWORDS = stopwords.DEFAULT
STOPWORDS = STOPWORDS.union(
    set([
        "article",
        "page",
        "wikipedia",
        "edit",
        "use",
        "one",
        "source",
        "like",
        "please",
        "thank",
        "good",
        "time",
        "link"
    ])
)

# custom pipelines
PIPELINE_CLEAN = [
    lambda s: th.fillna(s),
    lambda s: th.lowercase(s),
    lambda s: th.remove_digits(s, only_blocks=False),
    lambda s: th.remove_punctuation(s),
    lambda s: th.remove_brackets(s),
    lambda s: th.remove_diacritics(s),
    lambda s: th.remove_html_tags(s),
    lambda s: th.remove_urls(s),
    lambda s: th.remove_whitespace(s),
    lambda s: th.remove_stopwords(s, STOPWORDS),
    lambda s: th.stem(s) # snowball by default
]

PIPELINE_TOKENS = [
    lambda s: th.tokenize(s)
]

def clean_text(data, feature):
    """
    Clean text
    :param data: dataframe 
    :param feature: feature with text
    :returns: dataframe
    """

    clean = config.CLEANED_TEXT

    # get data
    input_filename = data + ".csv"
    df = pd.read_csv(os.path.join("../data", input_filename))

    # clean text
    df[clean] = df[feature].pipe(th.clean, PIPELINE_CLEAN)    
    # tokenize into new feature
    df['clean_tokens'] = df[feature].pipe(th.clean, PIPELINE_TOKENS)

    return df

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",   # name of data file
        type=str
    )
    parser.add_argument(
        "--feature",   # name of text feature
        type=str
    )

    args = parser.parse_args()

    # get preprocessed data
    cleaned = clean_text(args.data, args.feature)

    # store the file
    filename = args.data + "_clean.csv"
    cleaned.to_csv(os.path.join("../data", filename), index=False)

    print("Cleaned data stored in the `data` folder")