"""EASE: линейная модель с нулевой диагональю весов."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .base import Recommender


class EASE(Recommender):
    def __init__(self, regularization=300.0):
        self.regularization = regularization

    def fit(self, positive):
        if self.regularization <= 0:
            raise ValueError('Нужна положительная регуляризация')
        self.history = positive.astype(float).tocsr()
        gram = (self.history.T @ self.history).toarray(order='F')
        gram.flat[::gram.shape[0] + 1] += self.regularization
        factor = cho_factor(gram, overwrite_a=True, check_finite=False)
        inverse = cho_solve(factor, np.eye(gram.shape[0]), check_finite=False)
        diagonal = np.diag(inverse).copy()
        inverse /= -diagonal[None, :]
        np.fill_diagonal(inverse, 0)
        self.weights = inverse
        return self

    def scores(self, users):
        return self.history[users] @ self.weights
