import unittest

import numpy as np
from scipy import sparse

from experiment_utils import bootstrap_difference, configurations
from evaluation import recommend_all, ranking_metrics
from models.base import Recommender
from protocol import EvaluationSet


class ExperimentTests(unittest.TestCase):
    def test_search_space_and_als_control(self):
        names = ['ItemKNN', 'iALS', 'EASE', 'FM', 'EngagementALS']
        self.assertEqual([len(configurations(name)) for name in names], [4, 8, 4, 12, 8])
        self.assertEqual(configurations('iALS'), configurations('EngagementALS'))
        for name in names:
            self.assertEqual(len(configurations(name)), len({str(p) for p in configurations(name)}))

    def test_pair_bootstrap_known_difference(self):
        result = bootstrap_difference([0.3, 0.7, 0.9], [0.1, 0.5, 0.7], repeats=100)
        np.testing.assert_allclose(list(result.values()), [0.2, 0.2, 0.2])
        identical = bootstrap_difference([0, 1, 0.5], [0, 1, 0.5])
        self.assertEqual(identical, {'difference': 0.0, 'lower_95': 0.0, 'upper_95': 0.0})

    def test_recommendation_batches(self):
        class FixedModel(Recommender):
            def scores(self, users):
                return np.tile(np.arange(12), (len(users), 1))
        seen = sparse.csr_matrix(([1, 1], ([0, 1], [11, 0])), shape=(2, 12))
        relevant = sparse.csr_matrix(([1, 1, 1], ([0, 0, 1], [10, 9, 1])), shape=(2, 12))
        cohort = EvaluationSet(np.arange(2), seen, relevant)
        recommendations = recommend_all(FixedModel(), cohort, batch_size=1)
        np.testing.assert_array_equal(
            recommendations, recommend_all(FixedModel(), cohort, batch_size=2))
        metrics = ranking_metrics(recommendations, relevant)
        np.testing.assert_array_equal(recommendations[0], np.arange(10, 0, -1))
        self.assertEqual(metrics['NDCG@10'][0], 1)
        self.assertEqual(metrics['Recall@10'][1], 0)


if __name__ == '__main__':
    unittest.main()
