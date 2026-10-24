"""Независимые проверки формул, градиентов и общего протокола."""

import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from evaluation import evaluate, rank_scores, ranking_metrics
from features import build_item_features
from models import EASE, FM, IALS, EngagementALS, ItemKNN, engagement_target
from protocol import EvaluationSet, event_matrix, make_cohort, positive_matrix, split_and_encode


class RankingTests(unittest.TestCase):
    def test_metrics_by_hand(self):
        relevant = sparse.csr_matrix([[1, 0, 1, 0], [0, 1, 0, 0]])
        result = ranking_metrics(np.array([[0, 1, 2], [0, 2, 3]]), relevant)
        assert_allclose(result['Recall@3'], [1, 0])
        assert_allclose(result['NDCG@3'], [(1 + 0.5) / (1 + 1 / np.log2(3)), 0])
        perfect = ranking_metrics(np.array([[2, 0], [1, 3]]), relevant)
        assert_allclose(perfect['NDCG@2'], 1)

    def test_exclusion_ties_and_unique(self):
        scores = np.array([[4, 4, 4, 4], [1, 2, 3, 4]])
        seen = sparse.csr_matrix([[0, 1, 0, 0], [0, 0, 0, 1]])
        result = rank_scores(scores, seen, k=3)
        assert_array_equal(result, [[0, 2, 3], [2, 1, 0]])
        self.assertTrue(np.all([len(set(row)) == 3 for row in result]))
        with self.assertRaises(ValueError):
            rank_scores(scores, seen, k=4)
        self.assertEqual(scores[0, 1], 4)  # Входные оценки не изменяются.

    def test_macro_average_and_batching(self):
        class FixedModel:
            def recommend(self, users, seen, k):
                return np.tile([0, 1], (len(users), 1))
        cohort = EvaluationSet(np.array([0, 1]), sparse.csr_matrix((2, 4)),
                               sparse.csr_matrix([[1, 0, 0, 0], [0, 1, 1, 0]]))
        result = evaluate(FixedModel(), cohort, k=2, batch_size=1)
        self.assertEqual(result['Recall@2'], 0.75)
        self.assertEqual(result, evaluate(FixedModel(), cohort, k=2, batch_size=2))


class FormulaTests(unittest.TestCase):
    def setUp(self):
        self.x = sparse.csr_matrix([[1., 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 0]])

    def test_knn_cosine(self):
        model = ItemKNN(neighbors=3).fit(self.x)
        expected = np.zeros((4, 4))
        dense = self.x.toarray()
        for first in range(3):
            for second in range(3):
                if first != second:
                    expected[first, second] = np.dot(dense[:, first], dense[:, second]) / (
                        np.linalg.norm(dense[:, first]) * np.linalg.norm(dense[:, second]))
        assert_allclose(model.similarity.toarray(), expected)
        assert_allclose(model.scores([0, 2]), dense[[0, 2]] @ expected)
        self.assertLessEqual(ItemKNN(1).fit(self.x).similarity.getnnz(axis=1).max(), 1)

    def test_ease_separate_ridge_solutions(self):
        model = EASE(regularization=2).fit(self.x)
        x = self.x.toarray()
        expected = np.zeros((4, 4))
        for target in range(4):
            other = np.arange(4) != target
            design = x[:, other]
            expected[other, target] = np.linalg.solve(
                design.T @ design + 2 * np.eye(3), design.T @ x[:, target])
        assert_allclose(model.weights, expected, atol=1e-12)
        assert_allclose(model.scores([0, 1]), x[[0, 1]] @ expected)
        gradient = (x.T @ x + 2 * np.eye(4)) @ model.weights - x.T @ x
        np.fill_diagonal(gradient, 0)
        assert_allclose(gradient, 0, atol=1e-12)

    def test_als_update_with_observed_zero(self):
        targets = sparse.csr_matrix(([1., 0, 0.6], ([0, 0, 1], [0, 1, 2])), shape=(3, 4))
        fixed = np.array([[1., 2], [2, 0], [0, 1], [1, -1]])
        model = IALS(factors=2, alpha=4, regularization=0.7)
        result = model._update(targets, fixed)
        for user in range(3):
            confidence = np.ones(4)
            confidence[targets[user].indices] += 4
            expected = np.linalg.solve(
                fixed.T @ np.diag(confidence) @ fixed + 0.7 * np.eye(2),
                fixed.T @ (confidence * targets.toarray()[user]))
            assert_allclose(result[user], expected, atol=1e-12)
        without_zero = targets.copy()
        without_zero.eliminate_zeros()
        self.assertFalse(np.allclose(result, model._update(without_zero, fixed)))

    def test_als_objective_dense_and_descent(self):
        targets = sparse.csr_matrix(([1., 0, 1], ([0, 0, 1], [0, 1, 2])), shape=(3, 4))
        model = IALS(factors=2, alpha=4, regularization=0.7, iterations=4).fit(targets)
        confidence = np.ones(targets.shape)
        coo = targets.tocoo()
        confidence[coo.row, coo.col] += model.alpha
        predictions = model.user_factors @ model.item_factors.T
        expected = np.sum(confidence * (targets.toarray() - predictions) ** 2)
        expected += 0.7 * (np.sum(model.user_factors ** 2) + np.sum(model.item_factors ** 2))
        assert_allclose(model.objective(targets), expected, atol=1e-12)
        self.assertTrue(np.all(np.diff(model.loss_history) <= 1e-10))

    def test_custom_target_by_hand_and_same_solver(self):
        events = pd.DataFrame({
            'u': [0, 0, 1, 1, 2], 'i': [0, 1, 0, 1, 2],
            'play_time_ms': [0, 50, 200, 100, 100], 'duration_ms': [100] * 5,
            'is_like': [0, 0, 0, 1, 1], 'is_follow': [0, 0, 0, 1, 1],
            'is_hate': [0, 0, 0, 0, 1],
        })
        target = engagement_target(events)
        assert_allclose(target, [0, 0.35, 0.7, 1, 0])
        self.assertTrue(np.all((target >= 0) & (target <= 1)))
        matrix = event_matrix(events, (3, 3), target)
        self.assertEqual(matrix.nnz, len(events))
        custom = EngagementALS(factors=2, iterations=2).fit(events, (3, 3))
        plain = IALS(factors=2, iterations=2).fit(matrix)
        assert_array_equal(custom.user_factors, plain.user_factors)
        assert_array_equal(custom.item_factors, plain.item_factors)

    def test_als_reproducibility(self):
        first = IALS(factors=2, iterations=2).fit(self.x)
        second = IALS(factors=2, iterations=2).fit(self.x)
        assert_array_equal(first.scores([0, 1]), second.scores([0, 1]))


class FMTests(unittest.TestCase):
    def setUp(self):
        self.model = FM(factors=2, regularization=0.03)
        self.model.bias = np.array([0.2])
        self.model.weights = np.array([0.1, -0.2, 0.3, 0.4])
        self.model.vectors = np.arange(8).reshape(4, 2) * 0.04 - 0.1
        self.features = sparse.csr_matrix([[1., 0, 1, 0.7], [0, 1, 1, -0.2], [1, 0, 0, 0]])
        self.labels = np.array([1., 0, 1])

    def test_forward_explicit_pairs(self):
        expected = []
        for row in self.features.toarray():
            score = self.model.bias[0] + row @ self.model.weights
            for first in range(len(row)):
                for second in range(first + 1, len(row)):
                    score += row[first] * row[second] * np.dot(
                        self.model.vectors[first], self.model.vectors[second])
            expected.append(score)
        assert_allclose(self.model._forward(self.features), expected, atol=1e-12)

    def test_all_gradients_finite_difference(self):
        _, gradients = self.model._loss_and_grad(self.features, self.labels)
        for parameter, gradient in zip(
            [self.model.bias, self.model.weights, self.model.vectors], gradients,
        ):
            numerical = np.zeros_like(parameter)
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + 1e-6
                plus = self.model._loss_and_grad(self.features, self.labels)[0]
                parameter[index] = original - 1e-6
                minus = self.model._loss_and_grad(self.features, self.labels)[0]
                parameter[index] = original
                numerical[index] = (plus - minus) / 2e-6
            assert_allclose(gradient, numerical, atol=1e-8, rtol=1e-6)

    def test_cached_scores_match_general_forward(self):
        item_features = sparse.csr_matrix([[1., 0.7], [0, -0.2], [1, 0]])
        self.model._prepare_scoring(item_features, n_users=2)
        for user in range(2):
            user_features = sparse.csr_matrix(
                (np.ones(3), (np.arange(3), np.full(3, user))), shape=(3, 2))
            design = sparse.hstack([user_features, item_features], format='csr')
            assert_allclose(self.model.scores(np.array([user]))[0], self.model._forward(design))

    def test_training_and_reproducibility(self):
        users = np.tile([0, 0, 1, 1], 8)
        items = np.tile([0, 1, 0, 1], 8)
        labels = np.tile([1, 0, 0, 1], 8)
        arguments = (users, items, labels, sparse.eye(2, format='csr'), 2)
        first = FM(factors=3, epochs=12, batch_size=8).fit(*arguments)
        second = FM(factors=3, epochs=12, batch_size=8).fit(*arguments)
        self.assertLess(first.loss_history[-1], first.loss_history[0])
        assert_array_equal(first.scores(np.array([0, 1])), second.scores(np.array([0, 1])))
        self.assertEqual(first.loss_history, second.loss_history)


class ProtocolTests(unittest.TestCase):
    def test_boundaries_cold_start_and_history(self):
        events = pd.DataFrame({
            'user_id': [10, 20, 10, 10, 30, 10, 10, 10],
            'video_id': [1, 2, 3, 2, 1, 99, 3, 2],
            'is_relevant': [1, 0, 0, 1, 1, 1, 1, 1],
            'time_ms': [pd.Timestamp(date, tz='UTC').value // 10**6 for date in [
                '2022-04-21', '2022-04-21', '2022-04-21', '2022-04-22',
                '2022-04-23', '2022-04-23', '2022-05-01', '2022-05-02']],
        })
        parts, users, items = split_and_encode(events)
        assert_array_equal(users, [10, 20])
        assert_array_equal(items, [1, 2, 3])
        self.assertEqual([len(value) for value in parts.values()], [3, 3, 2])
        self.assertEqual((parts['validation'].u == -1).sum(), 1)
        self.assertEqual((parts['validation'].i == -1).sum(), 1)
        positives = positive_matrix(parts['train'], (2, 3))
        cohort = make_cohort(parts['validation'], parts['train'], positives, k=1)
        assert_array_equal(cohort.users, [0])
        self.assertEqual(cohort.seen[0, 2], 1)  # Исключён и отрицательный показ.
        history = pd.concat([parts['train'], parts['validation']])
        with self.assertRaises(ValueError):
            make_cohort(parts['test'], history, positives, k=1)  # Повторная пара.

    def test_features_fit_train_catalog_only(self):
        items = pd.DataFrame({'video_id': [1, 2, 99], 'tag': ['a,b', 'a', 'future'],
                              'video_duration': [100, 200, 999999]})
        features, info = build_item_features(items, np.array([1, 2]))
        self.assertEqual(info['tags'], 2)
        assert_allclose(features.toarray()[:, :2], np.eye(2))
        assert_allclose(features.toarray()[:, -1], [-1, 1])
        assert_allclose(features.toarray()[:, 2:4], [[1, 1], [1, 0]])


if __name__ == '__main__':
    unittest.main()
