"""
Feature selection script
------------------------

Contains elements to aid in feature selection
"""

from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile


class UnivariateFeatureSelection:
    """
    Univariate feature selection
    Choose any kind of scoring method
    """

    def __init__(self, n_features, scoring):
        """
        Univariate feature selection wrapper
        :param n_features: If this is float, it is SelectPercentile
            else it is SelectKBest
        :param scoring: The scoring function
        """

        # valid scoring functions
        valid_scoring = {
            "chi2": chi2,
            "f_classif": f_classif,
            "mutual_info_classif": mutual_info_classif
        }

        if isinstance(n_features, float):
            self.selection = SelectPercentile(
                valid_scoring[scoring],
                percentile = int(n_features * 100)
            )
        else:
            self.selection = SelectKBest(
                valid_scoring[scoring],
                k=n_features
            )
        
    def fit(self, X, y):
        return self.selection.fit(X, y)
    
    def transform(self, X):
        return self.selection.transform(X)

    def fit_transform(self, X, y):
        return self.selection.fit_transform(X, y)
        