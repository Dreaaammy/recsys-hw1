"""Признаки видео: ID, контентные метки и длительность."""

import numpy as np
from scipy import sparse


def build_item_features(items, train_item_ids):
    # Словарь и масштаб числового признака строятся только по каталогу train.
    catalog = items.set_index('video_id').loc[train_item_ids]
    tags = [sorted(set(str(value).split(','))) for value in catalog.tag]
    vocabulary = sorted({tag for row in tags for tag in row})
    tag_index = {tag: index for index, tag in enumerate(vocabulary)}
    rows, columns = [], []
    for item, item_tags in enumerate(tags):
        for tag in item_tags:
            rows.append(item)
            columns.append(tag_index[tag])
    tag_features = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, columns)),
        shape=(len(catalog), len(vocabulary)),
    )
    log_duration = np.log1p(catalog.video_duration.to_numpy())
    mean = log_duration.mean()
    scale = max(log_duration.std(), 1e-12)
    duration = (log_duration - mean) / scale
    features = sparse.hstack([
        sparse.eye(len(catalog), format='csr'),
        tag_features,
        sparse.csr_matrix(duration[:, None]),
    ], format='csr')
    info = {'tags': len(vocabulary), 'item_features': features.shape[1],
            'duration_log_mean': float(mean), 'duration_log_std': float(scale)}
    return features, info
