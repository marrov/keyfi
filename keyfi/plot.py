import warnings
import umap.plot
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from hdbscan import HDBSCAN, all_points_membership_vectors
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
    cmap = colors.ListedColormap(tuple(sns.color_palette('husl', n_clusters)))
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
    legend = ax.legend(handles=legend_elements, title='Clusters', fontsize=14,
                       title_fontsize=14, loc="lower left")
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((1, 1, 1, 0.25))


def _remove_axes(ax: plt.Subplot):
    ax.set(yticklabels=[], xticklabels=[])
    ax.tick_params(left=False, bottom=False)
    plt.tight_layout()


def _set_point_size(points: np.ndarray) -> np.ndarray:
    point_size = 100.0 / np.sqrt(points.shape[0])
    return point_size


def _set_cluster_member_colors(clusterer: HDBSCAN, soft: bool = True):
    n_clusters = np.size(np.unique(clusterer.labels_))

    if -1 in np.unique(clusterer.labels_) and not soft:
        color_palette = sns.color_palette('husl', n_clusters-1)
    else:
        color_palette = sns.color_palette('husl', n_clusters)

    if soft:
        soft_clusters = all_points_membership_vectors(clusterer)
        cluster_colors = [color_palette[np.argmax(x)]
                          for x in soft_clusters]
    else:
        cluster_colors = [color_palette[x] if x >= 0
                          else (0.5, 0.5, 0.5)
                          for x in clusterer.labels_]
    cluster_member_colors = [sns.desaturate(x, p)
                             for x, p
                             in zip(cluster_colors, clusterer.probabilities_)]
    return cluster_member_colors, color_palette


def _save_fig(save: bool = False, figname: str = None, figpath: str = None):
    if save:
        plt.savefig(figpath+figname+'.png')
    else:
        plt.show()


def plot_embedding(embedding: np.ndarray, data: pd.DataFrame = pd.DataFrame(), scale_points: bool = True, cmap_var: str = None, cmap_minmax: Sequence[Num] = list(), save: bool = False, figname: str = None, figpath: str = None):
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

    if scale_points:
        point_size = _set_point_size(embedding)
    else:
        point_size = None

    if cmap_var:
        if cmap_minmax:
            plt.scatter(*embedding.T, s=point_size, c=data[cmap_var],  vmin=cmap_minmax[0], vmax=cmap_minmax[1], cmap='inferno')
        else:
            plt.scatter(*embedding.T, s=point_size, c=data[cmap_var], cmap='inferno')
        _set_colorbar(label=cmap_var)
    else:
        plt.scatter(*embedding.T, s=point_size)

    _remove_axes(ax)
    _save_fig(save, figname, figpath)


def plot_clustering(embedding: np.ndarray, cluster_labels: np.ndarray, scale_points: bool = True, save: bool = False, figname: str = None, figpath: str = None):
    fig, ax = _set_plot_settings()
    n_clusters = np.size(np.unique(cluster_labels))

    if n_clusters > 30:
        warnings.warn(
            'Number of clusters (%s) too large, clustering visualization will be poor' % n_clusters)

    cmap, norm = _set_colors(n_clusters)

    if scale_points:
        point_size = _set_point_size(embedding)
    else:
        point_size = None

    plt.scatter(*embedding.T, s=point_size,
                c=cluster_labels, cmap=cmap, norm=norm)

    if n_clusters <= 12:
        _set_legend(labels=cluster_labels, cmap=cmap, ax=ax)
    else:
        _set_colorbar(label='Clusters', ticks=np.arange(n_clusters))
    
    _remove_axes(ax)
    _save_fig(save, figname, figpath)


def plot_cluster_membership(embedding: np.ndarray, clusterer: HDBSCAN, scale_points: bool = True, legend: bool = True, save: bool = False, figname: str = None, figpath: str = None, soft: bool = True):
    fig, ax = _set_plot_settings()

    n_clusters = np.size(np.unique(clusterer.labels_))

    if n_clusters > 12:
        warnings.warn(
            'Number of clusters (%s) too large, clustering visualization will be poor' % n_clusters)

    cluster_member_colors, color_palette = _set_cluster_member_colors(
        clusterer, soft)

    if scale_points:
        point_size = 5*_set_point_size(embedding)
    else:
        point_size = 20

    plt.scatter(*embedding.T, s=point_size, linewidth=0,
                c=cluster_member_colors, alpha=0.5)

    if legend:
        if -1 in np.unique(clusterer.labels_):
            unique_colors = ((0.5, 0.5, 0.5), *tuple(color_palette))
        else:
            unique_colors = tuple(color_palette)
        cmap = colors.ListedColormap(unique_colors)
        _set_legend(labels=clusterer.labels_, cmap=cmap, ax=ax)

    _remove_axes(ax)
    _save_fig(save, figname, figpath)


def umap_plot(mapper: Type, save: bool = False, figname: str = None, figpath: str = None, **kwargs):
    umap.plot.points(mapper, **kwargs)
    
    _save_fig(save, figname, figpath)

def plot_vtk_data(mesh, clusterer: HDBSCAN, legend: bool = True, soft: bool = True):
    import pyvista as pv
    cluster_member_colors, color_palette = _set_cluster_member_colors(clusterer, soft)

    labels=clusterer.labels_

    if legend:
        if -1 in np.unique(labels):
            unique_colors = ((0.5, 0.5, 0.5), *tuple(color_palette))
        else:
            unique_colors = tuple(color_palette)
        cmap = colors.ListedColormap(unique_colors)
    print (np.size(unique_colors))
    print (np.size(np.unique(clusterer.labels_))) #10000 labels from -1 to 10 -> -1 -1 -1 2 4 1 8
    #print (len(unique_colors)) #15 RGB colors
    #print (len(cluster_member_colors)) #10000
    #print (color_palette, len(color_palette)) # 14
    # Make a dictionary for the annotations
    annotations = {
    0.8: "High",
    805.3: "Cutoff value",
    }

    sargs = dict(
    mapper=None, 
    title_font_size=20,
    label_font_size=16,
    shadow=True,
    n_labels=len(unique_colors),
    italic=True,
    fmt="%.0f",
    font_family="arial",
    interactive=True,
    color='k'
    )  # Simply make the bar interactive
    unique_labels = np.unique(labels)
    print (unique_labels)
    #legend_elements = [Patch(facecolor=cmap.colors[i], label=unique_label)
    #for i, unique_label in enumerate(unique_labels)]
    #print (legend_elements)
    # First a default plot with jet colormap
    p = pv.Plotter(notebook=False)  # If in IPython, be sure to show the scene
    # Add the data, use active scalar for coloring, and show the scalar bar
    mesh['values'] = cluster_member_colors

    #mesh['values'] = labels
    unique_labels = np.unique(labels)
    #legend_elements = [Patch(facecolor=cmap.colors[i], label=unique_label)
                       #for i, unique_label in enumerate(unique_labels)]
    legend_elements = [[str(unique_label), cmap.colors[i]] for i, unique_label in enumerate(unique_labels)]
    print (legend_elements)
    p.add_mesh(mesh, scalar_bar_args=sargs, scalars='values', cmap=cmap)#, annotations=annotations) #clim=[0.7, 1.4]
    #legend_entries = []
    #legend_entries.append(['-1', (0.5,0.5,0.5)])
    #legend_entries.append(['0', 'k'])
    #p.add_legend(legend_entries)
    p.add_legend(legend_elements, size = [0.3, 0.3], origin = [0.1, 0.1])
    p.background_color = 'w'
    p.show()

    #mesh.plot(scalars='values', cmap=cmap)

    # Remove from plotters so output is not produced in docs
    pv.plotting._ALL_PLOTTERS.clear()
