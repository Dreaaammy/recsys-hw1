"""Независимый аудит test по исходным ID; запуск после experiments.ipynb."""

import csv
import gzip
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    train_end = int(datetime(2022, 4, 22, tzinfo=timezone.utc).timestamp() * 1000)
    validation_end = int(datetime(2022, 5, 1, tzinfo=timezone.utc).timestamp() * 1000)
    train_users, train_items, positive_users = set(), set(), set()
    seen, test_positive = defaultdict(set), defaultdict(set)
    with gzip.open(root / '.cache/prepared/kuairand/interactions.csv.gz', 'rt') as file:
        for row in csv.DictReader(file):
            user, item = int(row['user_id']), int(row['video_id'])
            time = int(row['time_ms'])
            positive = int(row['play_time_ms']) >= min(int(row['duration_ms']), 18_000)
            assert positive == int(row['is_relevant'])
            if time < train_end:
                train_users.add(user)
                train_items.add(item)
                if positive:
                    positive_users.add(user)
            if time < validation_end:
                seen[user].add(item)
            elif positive:
                test_positive[user].add(item)

    eligible = sorted(user for user in positive_users
                      if test_positive[user] & train_items
                      and len(train_items) - len(train_items & seen[user]) >= 10)
    saved = np.load(root / '.cache/results/test_recommendations.npz')
    raw_users = np.array(sorted(train_users))[saved['users']]
    np.testing.assert_array_equal(raw_users, eligible)
    np.testing.assert_array_equal(saved['item_ids'], sorted(train_items))
    with (root / 'reports/test_all_models.csv').open() as file:
        reported = {row['']: row for row in csv.DictReader(file)}

    differences = {}
    for name in reported:
        recommendations = saved['item_ids'][saved[name]]
        ndcg_values, recall_values = [], []
        for user, row in zip(raw_users, recommendations):
            truth = test_positive[user] & train_items
            assert len(set(row)) == 10 and not (set(row) & seen[user])
            dcg = sum(1 / math.log2(position + 2) for position, item in enumerate(row) if item in truth)
            ideal = sum(1 / math.log2(position + 2) for position in range(min(10, len(truth))))
            ndcg_values.append(dcg / ideal)
            recall_values.append(len(set(row) & truth) / len(truth))
        calculated = [sum(ndcg_values) / len(eligible), sum(recall_values) / len(eligible)]
        expected = [float(reported[name]['NDCG@10']), float(reported[name]['Recall@10'])]
        np.testing.assert_allclose(calculated, expected, rtol=0, atol=1e-12)
        differences[name] = float(np.max(np.abs(np.array(calculated) - expected)))

    report = {'method': 'csv + raw-ID sets + math; no protocol/evaluation imports',
              'test_users': len(eligible), 'catalog_size': len(train_items),
              'max_metric_differences': differences, 'all_recommendations_checked': True}
    (root / 'reports/checks.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
