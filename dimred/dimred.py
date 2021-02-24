import numpy as np
import pandas as pd

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import umap
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def import_csv_data(path: str = '') -> pd.DataFrame:
    '''
    Creates a pandas dataframe from the path to a csv data file
    '''
    if not path:
        path = input('Enter the path of your csv data file: ')
    return pd.read_csv(path)


def clean_data(data: pd.DataFrame, dim: int = 2):
    '''    
    Removes ghost cells (if present) and other data columns that
    are not relevant for the dimensionality reduction (i.e. spatial 
    coordinates) from the original data in place

    Parameters
    ----------
    dim : int
        Dimension of the dataset, use 2 for 2D-plane data and 3 
        for 3D-volume data   
    '''
    if dim not in [2, 3]:
        raise Exception(
            'dim can only be 2 or 3. Use 2 for 2D-plane data and 3 for 3D-volume data')

    cols_to_drop = ['XYZ:'+str(x) for x in range(3)]
    cols_to_drop.extend(['Phi', 'T'])

    if 'vtkGhostType' in data.columns:
        data.drop(data[data.vtkGhostType == 2].index, inplace=True)
        cols_to_drop.append('vtkGhostType')

    if dim == 2:
        cols_to_drop.append('U:2')

    data.drop(columns=cols_to_drop, axis=1, inplace=True)
    data.reset_index(drop=True, inplace=True)


def scale_data(data: pd.DataFrame) -> np.ndarray:
    scaled_data = StandardScaler().fit_transform(data)
    return scaled_data


def embed_data(data: pd.DataFrame, scale: bool = True) -> np.ndarray:
    if scale:
        data = scale_data(data)
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(data)
    return embedding


def main():
    path = 'data/LES/2D/2D_X_structured_subs5.csv'
    data = import_csv_data(path)
    print(data.head())
    clean_data(data, dim=2)
    print(data.head())
    embedding = embed_data(data, scale=True)

    # 6) Plot embedding
    fig, ax = plt.subplots(figsize=[6, 5])
    lfs = 18
    tfs = 16
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=data['N2O5_PPM'], vmin=0, vmax=20, cmap="inferno"
    )
    plt.xticks(fontsize=tfs)
    plt.yticks(fontsize=tfs)
    plt.gca().set_aspect('equal', 'datalim')
    cbaxes = inset_axes(ax, width="2.5%", height="50%",
                        loc='lower right', borderpad=0.5)
    cb = plt.colorbar(cax=cbaxes, orientation='vertical')
    cbaxes.yaxis.set_ticks_position('left')
    cb.ax.tick_params(labelsize=tfs)
    cb.set_label(r'N$_2$O$_5$ (ppm)', rotation=90, size=tfs, labelpad=-45)
    plt.show()


if __name__ == "__main__":
    main()
