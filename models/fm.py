"""FM второго порядка, бинарная кросс-энтропия и Adam на NumPy."""

import numpy as np
from scipy import sparse
from scipy.special import expit

from .base import Recommender


class FM(Recommender):
    def __init__(self, factors=16, learning_rate=0.01, regularization=1e-5,
                 epochs=3, batch_size=4096, seed=42):
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

    def _forward(self, features):
        projected = features @ self.vectors
        squared = features.multiply(features) @ (self.vectors ** 2)
        interactions = 0.5 * np.sum(projected ** 2 - squared, axis=1)
        return self.bias[0] + features @ self.weights + interactions

    def _loss_and_grad(self, features, labels):
        logits = self._forward(features)
        loss = np.mean(np.logaddexp(0, logits) - labels * logits)
        loss += 0.5 * self.regularization * (
            np.sum(self.weights ** 2) + np.sum(self.vectors ** 2)
        )
        error = (expit(logits) - labels) / len(labels)
        bias_grad = np.array([error.sum()])
        weight_grad = features.T @ error + self.regularization * self.weights
        projected = features @ self.vectors
        vector_grad = features.T @ (error[:, None] * projected)
        correction = features.multiply(features).T @ error
        vector_grad -= correction[:, None] * self.vectors
        vector_grad += self.regularization * self.vectors
        return float(loss), [bias_grad, weight_grad, vector_grad]

    def _training_loss(self, features, labels):
        total = 0.0
        for start in range(0, len(labels), self.batch_size):
            end = start + self.batch_size
            logits = self._forward(features[start:end])
            total += np.sum(np.logaddexp(0, logits) - labels[start:end] * logits)
        penalty = 0.5 * self.regularization * (
            np.sum(self.weights ** 2) + np.sum(self.vectors ** 2)
        )
        return float(total / len(labels) + penalty)

    def fit(self, users, items, labels, item_features, n_users):
        users, items = np.asarray(users), np.asarray(items)
        labels = np.asarray(labels, dtype=float)
        user_features = sparse.csr_matrix(
            (np.ones(len(users)), (np.arange(len(users)), users)),
            shape=(len(users), n_users),
        )
        features = sparse.hstack([user_features, item_features[items]], format='csr')
        rng = np.random.default_rng(self.seed)
        self.bias = np.zeros(1)
        self.weights = np.zeros(features.shape[1])
        self.vectors = rng.normal(0, 0.01, (features.shape[1], self.factors))
        parameters = [self.bias, self.weights, self.vectors]
        first_moments = [np.zeros_like(parameter) for parameter in parameters]
        second_moments = [np.zeros_like(parameter) for parameter in parameters]
        self.loss_history = [self._training_loss(features, labels)]
        step = 0
        for _ in range(self.epochs):
            order = rng.permutation(len(labels))
            for start in range(0, len(labels), self.batch_size):
                batch = order[start:start + self.batch_size]
                _, gradients = self._loss_and_grad(features[batch], labels[batch])
                step += 1
                for parameter, gradient, first, second in zip(
                    parameters, gradients, first_moments, second_moments,
                ):
                    first *= 0.9
                    first += 0.1 * gradient
                    second *= 0.999
                    second += 0.001 * gradient ** 2
                    corrected_first = first / (1 - 0.9 ** step)
                    corrected_second = second / (1 - 0.999 ** step)
                    parameter -= self.learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
            self.loss_history.append(self._training_loss(features, labels))
        self._prepare_scoring(item_features, n_users)
        return self

    def _prepare_scoring(self, item_features, n_users):
        item_vectors = self.vectors[n_users:]
        self.item_embeddings = item_features @ item_vectors
        squared = item_features.multiply(item_features) @ (item_vectors ** 2)
        self.item_offsets = item_features @ self.weights[n_users:]
        self.item_offsets += 0.5 * np.sum(self.item_embeddings ** 2 - squared, axis=1)

    def scores(self, users):
        # Точная FM-формула, сгруппированная по пользователю и видео.
        user_bias = self.bias[0] + self.weights[users]
        interactions = self.vectors[users] @ self.item_embeddings.T
        return user_bias[:, None] + self.item_offsets[None, :] + interactions
