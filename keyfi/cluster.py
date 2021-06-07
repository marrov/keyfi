from .plot import _set_cluster_member_colors, _save_fig

import numpy as np
import matplotlib.pyplot as plt

from typing import Type
from hdbscan import HDBSCAN
from matplotlib import colors
from sklearn.cluster import KMeans


def cluster_embedding(embedding: np.ndarray, algorithm, **params) -> Type:
    algorithms = [HDBSCAN, KMeans]
    if algorithm not in algorithms:
        raise ValueError(
            'invalid algorithm. Expected one of: %s' % algorithms)

    clusterer = algorithm(**params).fit(embedding)
    return clusterer


def show_condensed_tree(clusterer: HDBSCAN, select_clusters: bool = True, label_clusters: bool = True, leaf_separation: float = 0.5, save: bool = False, figname: str = None, figpath: str = None, **params):
    _, color_palette = _set_cluster_member_colors(clusterer)
    _, ax = plt.subplots(figsize=[12, 6])
    cmap=colors.LinearSegmentedColormap.from_list("", ["#bfc4c6", "#bfc4c6", "#bfc4c6"])
    
    clusterer.condensed_tree_.plot(axis=ax, cmap=cmap, leaf_separation=leaf_separation, select_clusters=select_clusters, label_clusters=label_clusters, selection_palette=color_palette[:], colorbar=False, **params)

    ax.set_ylabel('$\lambda$ value', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    _save_fig(save, figname, figpath)
