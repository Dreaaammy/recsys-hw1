"""ItemKNN: косинусная близость бинарных историй."""

import numpy as np
from scipy import sparse

from .base import Recommender


class ItemKNN(Recommender):
    def __init__(self, neighbors=50):
        self.neighbors = neighbors

    def fit(self, positive):
        self.history = positive.astype(float).tocsr()
        similarity = (self.history.T @ self.history).toarray()
        norms = np.sqrt(np.diag(similarity))
        denominator = np.outer(norms, norms)
        np.divide(similarity, denominator, out=similarity, where=denominator > 0)
        np.fill_diagonal(similarity, 0)
        rows, columns, values = [], [], []
        for item, row in enumerate(similarity):
            neighbors = np.argsort(-row, kind='stable')[:self.neighbors]
            neighbors = neighbors[row[neighbors] > 0]
            rows.extend([item] * len(neighbors))
            columns.extend(neighbors)
            values.extend(row[neighbors])
        self.similarity = sparse.csr_matrix(
            (values, (rows, columns)), shape=similarity.shape,
        )
        return self

    def scores(self, users):
        return (self.history[users] @ self.similarity).toarray()
