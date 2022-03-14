import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class MyEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Class that takes multiple trained binary classification models
    and predicts the outcome based on the average prediction probability
    of all models
    """

    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.where(self.predict_proba(X) >= 0.5, 1, 0)

    def predict_proba(self, X):
        probas = []
        for i, model in enumerate(self.models):
            pred = model.predict_proba(X)[:, 1]
            probas.append(pred)
        return np.array(probas).mean(axis=0)
