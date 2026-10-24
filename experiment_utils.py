"""Подбор параметров и интервалы для сравнения моделей."""

from itertools import product
import json

import numpy as np
import pandas as pd

from evaluation import evaluate
from models import EASE, FM, IALS, EngagementALS, ItemKNN
from protocol import event_matrix


SEARCH_SPACE = {
    'ItemKNN': {'neighbors': [20, 50, 100, 200]},
    'iALS': {'factors': [16, 32], 'alpha': [1, 10], 'regularization': [1, 10]},
    'EASE': {'regularization': [30, 100, 300, 1000]},
    'FM': {'factors': [16, 32], 'regularization': [1e-6, 1e-5, 1e-4], 'epochs': [3, 10]},
    'EngagementALS': {'factors': [16, 32], 'alpha': [1, 10], 'regularization': [1, 10]},
}
FIXED_PARAMETERS = {
    'ItemKNN': {},
    'iALS': {'iterations': 10, 'seed': 42},
    'EASE': {},
    'FM': {'learning_rate': 0.01, 'batch_size': 4096, 'seed': 42},
    'EngagementALS': {'iterations': 10, 'seed': 42},
}


def configurations(name):
    space = SEARCH_SPACE[name]
    result = []
    for values in product(*space.values()):
        parameters = FIXED_PARAMETERS[name].copy()
        parameters.update(zip(space, values))
        result.append(parameters)
    return result


def fit_model(name, parameters, train, positive, item_features):
    classes = {'ItemKNN': ItemKNN, 'iALS': IALS, 'EASE': EASE,
               'FM': FM, 'EngagementALS': EngagementALS}
    model = classes[name](**parameters)
    if name in ['ItemKNN', 'EASE']:
        model.fit(positive)
    elif name == 'iALS':
        targets = event_matrix(train, positive.shape, train.is_relevant.to_numpy())
        model.fit(targets)
    elif name == 'FM':
        model.fit(train.u, train.i, train.is_relevant, item_features, positive.shape[0])
    else:
        model.fit(train, positive.shape)
    return model


def search_models(train, positive, item_features, validation_set, report_dir):
    results, selected, best_models = [], {}, {}
    for name in SEARCH_SPACE:
        best_ndcg = -1.0
        for number, parameters in enumerate(configurations(name), start=1):
            model = fit_model(name, parameters, train, positive, item_features)
            metrics = evaluate(model, validation_set)
            row = {'model': name, 'trial': number, 'parameters': json.dumps(parameters, sort_keys=True),
                   **metrics}
            results.append(row)
            if metrics['NDCG@10'] > best_ndcg:
                best_ndcg = metrics['NDCG@10']
                selected[name] = parameters
                best_models[name] = model
        pd.DataFrame(results).to_csv(report_dir / 'validation_search.csv', index=False)
        print(f'{name}: {len(configurations(name))} настроек, NDCG@10 = {best_ndcg:.5f}', flush=True)
    # Настройки всех пяти моделей зафиксированы до первого расчёта test-метрик.
    (report_dir / 'selected_parameters.json').write_text(json.dumps(selected, indent=2))
    return pd.DataFrame(results), selected, best_models


def bootstrap_difference(first, second, repeats=2000, seed=42):
    """Парный bootstrap пользователей: разность средних first − second."""
    difference = np.asarray(first) - np.asarray(second)
    rng = np.random.default_rng(seed)
    means = np.empty(repeats)
    for iteration in range(repeats):
        sample = rng.integers(0, len(difference), size=len(difference))
        means[iteration] = difference[sample].mean()
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {'difference': float(difference.mean()), 'lower_95': float(lower),
            'upper_95': float(upper)}
