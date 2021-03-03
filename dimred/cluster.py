from dimred.plot import _set_colors

import hdbscan
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Type
from sklearn.cluster import KMeans


def cluster_embedding(embedding: np.ndarray, algorithm, **params) -> Type:
    algorithms = [hdbscan.HDBSCAN, KMeans]
    if algorithm not in algorithms:
        raise ValueError(
            'invalid algorithm. Expected one of: %s' % algorithms)

    clusterer = algorithm(**params).fit(embedding)
    return clusterer


def show_condensed_tree(clusterer: hdbscan.HDBSCAN, select_clusters: bool = True, label_clusters: bool = True, **params):
    n_clusters = np.size(np.unique(clusterer.labels_))
    cmap, _ = _set_colors(n_clusters)
    clusterer.condensed_tree_.plot(
        select_clusters=select_clusters,
        selection_palette=list(cmap.colors),
        **params
    )
    plt.show()
