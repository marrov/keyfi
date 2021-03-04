from dimred.plot import plot_embedding, plot_clustering, umap_plot
from dimred.cluster import cluster_embedding, show_condensed_tree

import umap
import hdbscan
import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Union, Sequence, Tuple, Type


def import_csv_data(path: str = '') -> pd.DataFrame:
    '''
    Creates a pandas dataframe from path to a csv data file.
    '''
    if not path:
        path = input('Enter the path of your csv data file: ')
    return pd.read_csv(path)


def clean_data(data: pd.DataFrame, dim: int = 2):
    '''    
    Removes ghost cells (if present) and other data columns that
    are not relevant for the dimensionality reduction (i.e. spatial 
    coordinates) from the original data in place .
    '''
    if dim not in [2, 3]:
        raise ValueError(
            'dim can only be 2 or 3. Use 2 for 2D-plane data and 3 for 3D-volume data')

    cols_to_drop = ['XYZ:'+str(x) for x in range(3)]
    cols_to_drop.extend(['T'])

    if 'vtkGhostType' in data.columns:
        data.drop(data[data.vtkGhostType == 2].index, inplace=True)
        cols_to_drop.append('vtkGhostType')

    if dim == 2:
        cols_to_drop.append('U:2')

    data.drop(columns=cols_to_drop, axis=1, inplace=True)
    data.reset_index(drop=True, inplace=True)


def scale_data(data: pd.DataFrame) -> np.ndarray:
    '''    
    Scales input data based on sklearn standard scaler.
    '''
    scaled_data = StandardScaler().fit_transform(data)
    return scaled_data


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


def main():

    path = 'data/LES/2D/toy.csv'

    data = import_csv_data(path)

    clean_data(data, dim=2)

    embedding, mapper = embed_data(
        data=data,
        algorithm=umap.UMAP,
        scale=True,
        n_neighbors=20,
        min_dist=0.2,
    )

    clusterer = cluster_embedding(
        embedding=embedding,
        algorithm=hdbscan.HDBSCAN,
        min_cluster_size=25
    )

    plot_clustering(
        embedding=embedding,
        cluster_labels=clusterer.labels_,
        use_legend=True
    )

    # Useful code:
    #
    # plot_embedding(embedding=embedding, data=data, scale_points=True, cmap_var='Phi', cmap_minmax=[0, 5])
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


if __name__ == '__main__':
    main()
