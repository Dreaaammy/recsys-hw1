"""ALS для неявных данных с повышенным весом наблюдавшихся показов."""

import numpy as np
from scipy.linalg import solve

from .base import Recommender


class IALS(Recommender):
    def __init__(self, factors=16, alpha=10.0, regularization=1.0,
                 iterations=5, seed=42):
        self.factors = factors
        self.alpha = alpha
        self.regularization = regularization
        self.iterations = iterations
        self.seed = seed

    def _update(self, targets, fixed):
        base = fixed.T @ fixed + self.regularization * np.eye(self.factors)
        result = np.zeros((targets.shape[0], self.factors))
        for row in range(targets.shape[0]):
            start, end = targets.indptr[row:row + 2]
            vectors = fixed[targets.indices[start:end]]
            values = targets.data[start:end]
            matrix = base + self.alpha * (vectors.T @ vectors)
            right = (1 + self.alpha) * (vectors.T @ values)
            result[row] = solve(matrix, right, assume_a='pos', check_finite=False)
        return result

    def objective(self, targets):
        users, items = self.user_factors, self.item_factors
        # Базовый вес 1 для всех пар, затем поправка для наблюдавшихся.
        loss = np.sum((users.T @ users) * (items.T @ items))
        coo = targets.tocoo()
        for start in range(0, len(coo.data), 100_000):
            end = start + 100_000
            predictions = np.sum(users[coo.row[start:end]] * items[coo.col[start:end]], axis=1)
            errors = coo.data[start:end] - predictions
            loss += np.sum((1 + self.alpha) * errors ** 2 - predictions ** 2)
        loss += self.regularization * (np.sum(users ** 2) + np.sum(items ** 2))
        return float(loss)

    def fit(self, targets):
        targets = targets.astype(float).tocsr()
        if not np.isfinite(targets.data).all() or np.any(targets.data < 0):
            raise ValueError('Таргеты должны быть конечными и неотрицательными')
        if self.regularization <= 0 or self.alpha < 0:
            raise ValueError('Нужны regularization > 0 и alpha >= 0')
        # Не удаляем явные нули: неудачный показ отличается от отсутствия показа.
        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0, 0.01, (targets.shape[0], self.factors))
        self.item_factors = rng.normal(0, 0.01, (targets.shape[1], self.factors))
        self.loss_history = [self.objective(targets)]
        item_targets = targets.T.tocsr()
        for _ in range(self.iterations):
            self.user_factors = self._update(targets, self.item_factors)
            self.item_factors = self._update(item_targets, self.user_factors)
            self.loss_history.append(self.objective(targets))
        return self

    def scores(self, users):
        return self.user_factors[users] @ self.item_factors.T
