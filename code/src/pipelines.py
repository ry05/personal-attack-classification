"""
Text Pipelines
--------------

The use of a `pipeline` is to chain several data operations together and
automate some or most parts of the ML process
Advantages of using a pipeline
1. Convenience
2. Prevents leakage
3. Easier hyperparameter tuning
Source: https://scikit-learn.org/stable/modules/compose.html

All pipelines in this file set after performing hyperparameter optimization
The code for hyperparameter optimization for the best model
is presented in `ml_best_hyp_opt.py`
"""

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from fastai.text.all import *

from preprocess import StemTokenizer
from feature_selection import UnivariateFeatureSelection

# fastai's tokenizer function
spacy = WordTokenizer()
tokenizer = Tokenizer(spacy)

# pipeline 1
pipeline_1 = Pipeline([
    ('countvec', CountVectorizer()),
    ('clf', MultinomialNB())
])

# pipeline 2
pipeline_2 = Pipeline([
    ('countvec', CountVectorizer(
        stop_words = 'english'
    )),
    ('clf', ComplementNB())
])

# pipeline 3
pipeline_3 = Pipeline([
    ('tfidf', TfidfVectorizer(
        tokenizer = StemTokenizer(),
        # tokenizer = tokenizer,
        # token_pattern = None,
        # lowercase = False,
        max_features=10000
    )),
    ('ufs', UnivariateFeatureSelection(
        n_features = 0.05,     # Top 5% of the features built
        scoring = 'chi2'     
    )),
    ('clf', ComplementNB(alpha=0.01))     # classifier
])

# pipeline 4
numeric_transformer = Pipeline([
    ('scaler', MinMaxScaler())
])

# transformer for text feature
# use string as vectorizer converts a single vector into multiple vectors
text_transformer = Pipeline([
    ('tfidf', TfidfVectorizer(
        # tokenizer = StemTokenizer(),
        # token_pattern = None,
        tokenizer = tokenizer,
        # lowercase = False,
        max_features=10000
    )),
    ('ufs', UnivariateFeatureSelection(
        n_features = 0.05,     # Top 5% of the features built
        scoring = 'chi2'     
    ))
])

# hardcoded
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
pipeline_4 = Pipeline([
    ('preprocess', preprocessor),
    ('clf', ComplementNB(alpha=0.5)) # ComplementNB was used
])


# pipeline dictionary
pipe_dict = {
    '1': pipeline_1,
    '2': pipeline_2,
    '3': pipeline_3,
    '4': pipeline_4
}