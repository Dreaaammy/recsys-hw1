"""Общие правила рекомендаций и оценки для всех моделей."""

import numpy as np


def rank_scores(scores, seen, k=10):
    scores = np.array(scores, dtype=float, copy=True)
    if scores.shape != seen.shape or not np.isfinite(scores).all():
        raise ValueError('Некорректная матрица оценок')
    if k < 1 or np.any(scores.shape[1] - seen.getnnz(axis=1) < k):
        raise ValueError('Недостаточно кандидатов')
    rows, columns = seen.nonzero()
    scores[rows, columns] = -np.inf
    # При равенстве оценок первым идёт меньший индекс видео.
    return np.argsort(-scores, axis=1, kind='stable')[:, :k]


def ranking_metrics(recommendations, relevant):
    """Recall и NDCG для каждого пользователя; релевантность бинарная."""
    k = recommendations.shape[1]
    counts = relevant.getnnz(axis=1)
    if np.any(counts == 0):
        raise ValueError('Для оценки нужен хотя бы один релевантный объект')
    hits = np.zeros(recommendations.shape)
    for user, recommended in enumerate(recommendations):
        hits[user] = np.isin(recommended, relevant[user].indices)
    discounts = 1 / np.log2(np.arange(2, k + 2))
    recall = hits.sum(axis=1) / counts
    dcg = hits @ discounts
    ideal_dcg = np.cumsum(discounts)[np.minimum(counts, k) - 1]
    return {'NDCG@' + str(k): dcg / ideal_dcg,
            'Recall@' + str(k): recall}


def recommend_all(model, cohort, k=10, batch_size=128):
    if len(cohort.users) == 0:
        raise ValueError('Пустая группа оценки')
    recommendations = np.empty((len(cohort.users), k), dtype=int)
    for start in range(0, len(cohort.users), batch_size):
        users = cohort.users[start:start + batch_size]
        recommendations[start:start + len(users)] = model.recommend(users, cohort.seen, k)
    return recommendations


def evaluate(model, cohort, k=10, batch_size=128):
    recommendations = recommend_all(model, cohort, k, batch_size)
    values = ranking_metrics(recommendations, cohort.relevant[cohort.users])
    return {name: float(scores.mean()) for name, scores in values.items()}
