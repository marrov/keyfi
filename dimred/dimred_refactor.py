#from dimred.plot import plot_embedding, plot_clustering, plot_cluster_membership, umap_plot
#from dimred.cluster import cluster_embedding, show_condensed_tree
#
#import time
#
#import hdbscan
import numpy as np
import pandas as pd
#
#from sklearn.cluster import KMeans
#from sklearn.preprocessing import StandardScaler
from typing import Sequence, Tuple, Type


def import_csv_data(path: str = '') -> pd.DataFrame:
    '''
    Creates a pandas dataframe from path to a csv data file.
    '''
    if not path:
        path = input('Enter the path of your csv data file: ')
    return pd.read_csv(path)


def import_vtk_data(path: str = '') -> pd.DataFrame:
    '''
    Creates a pandas dataframe from path to a vtk data file.
    Also returns mesh pyvista object.
    '''
    import pyvista as pv

    if not path:
        path = input('Enter the path of your csv data file: ')

    mesh = pv.read(path)
    # Include undesired variables and vectors
    var_names_to_drop = ['U', 'vtkGhostType']
    var_names = [name for name in mesh.array_names if name not in var_names_to_drop]
    var_arrays = np.transpose([mesh.get_array(var_name) for var_name in var_names])
    df = pd.DataFrame(var_arrays, columns=var_names)
    # Add the velocity back with one row per component
    df[['U:0', 'U:1', 'U:2']] = mesh.get_array('U')
    return df, mesh


def export_vtk_data(mesh: Type, path: str = '', cluster_labels: np.ndarray = None):
    '''
    Exports vtk file with mesh. If cluster labels are passed it
    will include them in a new variable
    '''
    if cluster_labels is not None:
        mesh['clusters'] = cluster_labels
    mesh.save(path)

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


def scale_data(data: pd.DataFrame) -> np.ndarray:
    '''    
    Scales input data based on sklearn standard scaler.
    '''
    scaled_data = StandardScaler().fit_transform(data)
    return scaled_data


def embed_data(data: pd.DataFrame, algorithm: str = 'UMAP', scale: bool = True, **params) -> Tuple[np.ndarray, Type]:
    '''
    Applies either UMAP or t-SNE dimensionality reduction algorithm 
    to the input data (with optional scaling) and returns the
    embedding array. Also accepts specific and optional algorithm 
    parameters.
    '''
    algorithms = ['umap', 'tsne']
    if algorithm.lower() not in algorithms:
        raise ValueError(
            'invalid algorithm. Expected one of: %s' % algorithms)

    if scale:
        data = scale_data(data)

    if algorithm.lower() == 'umap':
        import umap
        reducer = umap.UMAP(**params)
        mapper = reducer.fit(data)
        embedding = mapper.transform(data)
    elif algorithm.lower() == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(**params)
        mapper = None
        embedding = reducer.fit_transform(data)

    return embedding, mapper


def main():

    path_input = 'data/input/2D_212_35.vtk'
    path_output = 'data/output/2D_212_35.vtk'

    data, mesh = import_vtk_data(path_input)
    cleaned_data = clean_data(data, dim=2, vars_to_drop=['U:0', 'U:1', 'U:2'])

    embedding, mapper = embed_data(
        data=cleaned_data,
        algorithm='tsne',
        scale=False,
        #n_neighbors=20,
        #min_dist=0.3,
    )

    print(f'{embedding}')
#
#    
#    clusterer = cluster_embedding(
#        embedding=embedding,
#        algorithm=hdbscan.HDBSCAN,
#        min_cluster_size=60
#    )
#
#    export_vtk_data(mesh=mesh, path=path_output, cluster_labels=clusterer.labels_)
#
#    plot_cluster_membership(embedding=embedding, clusterer=clusterer, save=False)

    # Other useful code snippets:
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


if __name__ == '__main__':
    main()
