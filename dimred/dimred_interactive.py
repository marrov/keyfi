# %% Import packages

import time
import umap
import hdbscan
import warnings
import umap.plot
import numpy as np
import pandas as pd
import pyvista as pv
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import colors
from matplotlib import rcParams
from matplotlib.patches import Patch

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Union, Sequence, Tuple, Type


# %% Time measure decorator
from functools import wraps
from time import process_time


def measure(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = int(round(process_time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(process_time() * 1000)) - start
            print(
                f"Execution time {func.__name__}: {end_ if end_ > 0 else 0} ms"
            )
    return _time_it

# %% Declare dimred functions

@measure
def import_csv_data(path: str = '') -> pd.DataFrame:
    '''
    Creates a pandas dataframe from path to a csv data file.
    '''
    if not path:
        path = input('Enter the path of your csv data file: ')
    return pd.read_csv(path)

@measure
def import_vtk_data(path: str = '') -> pd.DataFrame:
    '''
    Creates a pandas dataframe from path to a vtk data file.
    Also returns mesh pyvista object.
    '''
    if not path:
        path = input('Enter the path of your csv data file: ')

    mesh = pv.read(path)
    # Include undesired variables and vectors
    var_names_to_drop = ['U', 'vtkGhostType']

    if type(mesh) == pv.MultiBlock:
        mesh = mesh.get(0)
        var_names_to_drop.append('TimeValue')

    var_names = [
        name for name in mesh.array_names if name not in var_names_to_drop]
    var_arrays = np.transpose([mesh.get_array(var_name)
                               for var_name in var_names])
    df = pd.DataFrame(var_arrays, columns=var_names)
    # Add the velocity back with one row per component
    df[['U:0', 'U:1', 'U:2']] = mesh.get_array('U')
    return df, mesh

@measure
def export_vtk_data(mesh: Type, path: str = '', cluster_labels: np.ndarray = None):
    '''
    Exports vtk file with mesh. If cluster labels are passed it
    will include them in a new variable
    '''
    if cluster_labels is not None:
        mesh['clusters'] = cluster_labels
    mesh.save(path)

@measure
def clean_data(data: pd.DataFrame, dim: int = 2, vars_to_drop: Sequence[str] = None) -> pd.DataFrame:
    '''    
    Removes ghost cells (if present) and other data columns that
    are not relevant for the dimensionality reduction (i.e. spatial 
    coordinates) from the original data.
    '''
    if dim not in [2, 3]:
        raise ValueError(
            'dim can only be 2 or 3. Use 2 for 2D-plane data and 3 for 3D-volume data')

    cols_to_drop = []

    if 'Points:0' in data.columns:
        cols_to_drop.append(['Points:0', 'Points:1', 'Points:2'])

    if 'vtkGhostType' in data.columns:
        data.drop(data[data.vtkGhostType == 2].index, inplace=True)
        cols_to_drop.append('vtkGhostType')

    if 'U:0' in data.columns and dim == 2:
        cols_to_drop.append('U:2')

    if vars_to_drop is not None:
        cols_to_drop.extend(vars_to_drop)

    cleaned_data = data.drop(columns=cols_to_drop, axis=1)
    cleaned_data.reset_index(drop=True, inplace=True)

    return cleaned_data

@measure
def scale_data(data: pd.DataFrame) -> np.ndarray:
    '''    
    Scales input data based on sklearn standard scaler.
    '''
    scaled_data = StandardScaler().fit_transform(data)
    return scaled_data

@measure
def embed_data(data: pd.DataFrame, algorithm, scale: bool = True, **params) -> Tuple[np.ndarray, Type]:
    '''
    Applies either UMAP or t-SNE dimensionality reduction algorithm 
    to the input data (with optional scaling) and returns the
    embedding array. Also accepts specific and optional algorithm 
    parameters.
    '''
    algorithms = [umap.UMAP, TSNE]
    if algorithm not in algorithms:
        raise ValueError(
            'invalid algorithm. Expected one of: %s' % algorithms)

    if scale:
        data = scale_data(data)

    reducer = algorithm(**params)

    if algorithm == umap.UMAP:
        mapper = reducer.fit(data)
        embedding = mapper.transform(data)
    elif algorithm == TSNE:
        mapper = None
        embedding = reducer.fit_transform(data)

    return embedding, mapper

# %% Declare cluster functions

@measure
def cluster_embedding(embedding: np.ndarray, algorithm, **params) -> Type:
    algorithms = [hdbscan.HDBSCAN, KMeans]
    if algorithm not in algorithms:
        raise ValueError(
            'invalid algorithm. Expected one of: %s' % algorithms)

    clusterer = algorithm(**params).fit(embedding)
    return clusterer

@measure
def show_condensed_tree(clusterer: hdbscan.HDBSCAN, select_clusters: bool = True, label_clusters: bool = True, **params):
    n_clusters = np.size(np.unique(clusterer.labels_))
    cmap, _ = _set_colors(n_clusters)
    clusterer.condensed_tree_.plot(
        select_clusters=select_clusters,
        selection_palette=list(cmap.colors),
        **params
    )
    plt.show()

# %% Declare plot functions


Num = Union[int, float]
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'

@measure
def _set_plot_settings() -> Tuple[plt.Figure, plt.Subplot]:
    fig, ax = plt.subplots(figsize=[6, 5])
    plt.gca().set_aspect('equal', 'datalim')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    return fig, ax

@measure
def _set_colors(n_clusters: int) -> Tuple[colors.ListedColormap, colors.BoundaryNorm]:
    cmap = colors.ListedColormap(tuple(sns.color_palette('husl', n_clusters)))
    norm = colors.BoundaryNorm(np.arange(-0.5, n_clusters), n_clusters)
    return cmap, norm

@measure
def _set_colorbar(label: str = None, **kwargs):
    cb = plt.colorbar(**kwargs)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label, size=16)

@measure
def _set_legend(labels: np.ndarray, cmap: colors.ListedColormap, ax: plt.Subplot):
    unique_labels = np.unique(labels)
    legend_elements = [Patch(facecolor=cmap.colors[i], label=unique_label)
                       for i, unique_label in enumerate(unique_labels)]
    legend = ax.legend(handles=legend_elements, title='Clusters', fontsize=14,
                       title_fontsize=14, loc="upper right")
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((1, 1, 1, 0.25))

@measure
def _remove_axes(ax: plt.Subplot):
    ax.set(yticklabels=[], xticklabels=[])
    ax.tick_params(left=False, bottom=False)

@measure
def _set_point_size(points: np.ndarray) -> np.ndarray:
    point_size = 100.0 / np.sqrt(points.shape[0])
    return point_size

@measure
def _set_cluster_member_colors(clusterer: hdbscan.HDBSCAN):
    n_clusters = np.size(np.unique(clusterer.labels_))
    if -1 in np.unique(clusterer.labels_):
        color_palette = sns.color_palette('husl', n_clusters-1)
    else:
        color_palette = sns.color_palette('husl', n_clusters)
    cluster_colors = [color_palette[x] if x >= 0
                      else (0.5, 0.5, 0.5)
                      for x in clusterer.labels_]
    cluster_member_colors = [sns.desaturate(x, p)
                             for x, p
                             in zip(cluster_colors, clusterer.probabilities_)]
    return cluster_member_colors, color_palette

@measure
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
            plt.scatter(*embedding.T, s=point_size,
                        c=data[cmap_var],  vmin=cmap_minmax[0], vmax=cmap_minmax[1], cmap='inferno')
        else:
            plt.scatter(*embedding.T, s=point_size,
                        c=data[cmap_var], cmap='inferno')
        _set_colorbar(label=cmap_var)
    else:
        plt.scatter(*embedding.T, s=point_size)

    _remove_axes(ax)
    plt.tight_layout()

    if save:
        plt.savefig(figpath+figname+'.png')
    else:
        plt.show()

@measure
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
    plt.tight_layout()

    if save:
        plt.savefig(figpath+figname+'.png')
    else:
        plt.show()

@measure
def plot_cluster_membership(embedding: np.ndarray, clusterer: hdbscan.HDBSCAN, scale_points: bool = True, legend: bool = True, save: bool = False, figname: str = None, figpath: str = None):
    fig, ax = _set_plot_settings()

    cluster_member_colors, color_palette = _set_cluster_member_colors(
        clusterer)

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
    plt.tight_layout()

    if save:
        plt.savefig(figpath+figname+'.png')
    else:
        plt.show()

@measure
def umap_plot(mapper: Type, save: bool = False, figname: str = None, figpath: str = None, **kwargs):
    umap.plot.points(mapper, **kwargs)

    if save:
        plt.savefig(figpath+figname+'.png')
    else:
        plt.show()

# %% Read mesh

path_input = '../data/input/2D_848_140.vtk'
data, mesh = import_vtk_data(path_input)

# %% Clean data

cleaned_data = clean_data(data, dim=2)

# %% Compute embedding

embedding, mapper = embed_data(
    data=cleaned_data,
    #algorithm=umap.UMAP,
    #scale=True,
    #n_neighbors=20,
    #min_dist=0.2,
    algorithm=TSNE,
    scale=True,
    perplexity=40
)

# %% Compute clustering

clusterer = cluster_embedding(
    embedding=embedding,
    algorithm=hdbscan.HDBSCAN,
    min_cluster_size=150
)

# %% Export clusters as VTK

path_output = '../data/output/clusters.vtk'
export_vtk_data(mesh=mesh, path=path_output, cluster_labels=clusterer.labels_)

# %% Plot cluster membership

plot_cluster_membership(embedding=embedding, clusterer=clusterer, save=False)


# %% Other useful code snippets:
#
# plot_embedding(embedding=embedding, data=data, scale_points=True, cmap_var='Phi', cmap_minmax=[0, 5])
#
# plot_clustering(
#     embedding=embedding,
#     cluster_labels=clusterer.labels_
# )
#
# embedding, mapper = embed_data(
#     data=cleaned_data,
#     algorithm=TSNE,
#     scale=True,
#     perplexity=40
# )
#
# clusterer = clustering(
#    embedding=embedding,
#    algorithm=KMeans,
#    n_clusters=3,
#    init='k-means++',
#    max_iter=300,
#    n_init=10
#    )
#
# umap_plot(mapper, labels=clusterer.labels_)
#
# show_condensed_tree(
#     clusterer,
#     select_clusters=True,
#     label_clusters=True,
#     log_size=True
# )
