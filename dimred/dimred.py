from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import umap
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def main():
    # Note: WIP
    # This is a very initial implementation to sketch the workflow
    # 0) Define path to data
    path = 'data/LES/2D/2D_X_structured_subs5.csv'
    # 1) Import data
    data = pd.read_csv(path)
    # 2) Clean data
    data.drop(data[data.vtkGhostType == 2].index, inplace=True)
    data.drop(['vtkGhostType'], axis=1, inplace=True)
    data.reset_index(drop=True, inplace=True)
    # 3) Generate reducer object
    reducer = umap.UMAP()
    # 4) Scale data
    scaled_data = StandardScaler().fit_transform(data)
    # 5) Run dimensionality reduction
    embedding = reducer.fit_transform(scaled_data)
    # 6) Plot embedding
    fig, ax = plt.subplots(figsize=[8, 6])
    cm = plt.cm.get_cmap('inferno')
    lfs = 18
    tfs = 16
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=data['Phi'], vmin=0, vmax=5,
    )
    plt.xticks(fontsize=tfs)
    plt.yticks(fontsize=tfs)
    plt.gca().set_aspect('equal', 'datalim')
    cbaxes = inset_axes(ax, width="5%", height="60%",
                        loc='upper right', borderpad=1)
    cb = plt.colorbar(cax=cbaxes, orientation='vertical',
                      ticks=[0, 1, 2, 3, 4, 5])
    cbaxes.yaxis.set_ticks_position('left')
    cb.ax.tick_params(labelsize=tfs)
    cb.set_label(r'$\Phi$', rotation=180, size=lfs, labelpad=-50)
    cb.ax.set_yticklabels(['0', '1', '2', '3', '4', '< 5'])
    plt.show()
