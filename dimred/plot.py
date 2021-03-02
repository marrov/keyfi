import umap.plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import rcParams
from matplotlib.patches import Patch
from typing import Union, Sequence, Tuple, Type

Num = Union[int, float]
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'


def _set_plot_settings() -> Tuple[plt.Figure, plt.Subplot]:
    fig, ax = plt.subplots(figsize=[6, 5])
    plt.gca().set_aspect('equal', 'datalim')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    return fig, ax


def _set_colors(n_clusters: int) -> Tuple[colors.ListedColormap, colors.BoundaryNorm]:
    cmap_orig = plt.cm.get_cmap('tab10')
    if n_clusters > cmap_orig.N:
        raise ValueError(
            'number of clusters cannot be higher than number of available colors.')
    cmap = colors.ListedColormap(cmap_orig.colors[0:n_clusters])
    norm = colors.BoundaryNorm(np.arange(-0.5, n_clusters), n_clusters)
    return cmap, norm


def _set_colorbar(label: str = None, **kwargs):
    cb = plt.colorbar(**kwargs)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label, size=16)


def _set_legend(labels: np.ndarray, cmap: colors.ListedColormap, ax: plt.Subplot):
    unique_labels = np.unique(labels)
    legend_elements = [Patch(facecolor=cmap.colors[i], label=unique_label)
                       for i, unique_label in enumerate(unique_labels)]
    legend = ax.legend(handles=legend_elements, title='Clusters', fontsize=16,
                       title_fontsize=16, loc="upper right")
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((1, 1, 1, 0.25))


def _remove_axes(ax: plt.Subplot):
    ax.set(yticklabels=[], xticklabels=[])
    ax.tick_params(left=False, bottom=False)


def plot_embedding(embedding: np.ndarray, data: pd.DataFrame = pd.DataFrame(), cmap_var: str = None, cmap_minmax: Sequence[Num] = list()):
    '''
    Plots input embedding as a scatter plot. Optionally, a variable
    with an optional range can be supplied for use in the colormap.
    '''
    if cmap_var not in data.columns and cmap_var:
        raise ValueError(
            'invalid variable for the color map. Expected one of: %s' % data.columns)

    if len(cmap_minmax) != 2 and cmap_minmax:
        raise ValueError(
            'too many values to unpack. Expected 2')

    fig, ax = _set_plot_settings()

    if cmap_var:
        if cmap_minmax:
            plt.scatter(embedding[:, 0], embedding[:, 1], c=data[cmap_var],
                        vmin=cmap_minmax[0], vmax=cmap_minmax[1], cmap='inferno')
        else:
            plt.scatter(embedding[:, 0], embedding[:, 1],
                        c=data[cmap_var], cmap='inferno')
        _set_colorbar(label=cmap_var)
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1])

    _remove_axes(ax)
    plt.tight_layout()
    plt.show()


def plot_clustering(embedding: np.ndarray, cluster_labels: np.ndarray, use_legend: bool = True, scale_points: bool = True):
    fig, ax = _set_plot_settings()
    n_clusters = np.size(np.unique(cluster_labels))
    cmap, norm = _set_colors(n_clusters)

    point_size = {}
    if scale_points:
        point_size = 100.0 / np.sqrt(embedding.shape[0])

    plt.scatter(embedding[:, 0], embedding[:, 1], s=point_size,
                c=cluster_labels, cmap=cmap, norm=norm)

    if use_legend:
        _set_legend(labels=cluster_labels, cmap=cmap, ax=ax)
    else:
        _set_colorbar(label='Clusters', ticks=np.arange(n_clusters))
    _remove_axes(ax)
    plt.tight_layout()
    plt.show()


def umap_plot(mapper: Type, **kwargs):
    umap.plot.points(mapper, **kwargs)
    plt.show()
