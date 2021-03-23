# %% Import modules

# Add to path so that dimred is visible to this script
import sys
sys.path.append('../../')

# Import local functions from dimred
from dimred.dimred import *

# %% Define function to run UMAP with variable parameters

def run_UMAP(data, figname, **params):
    cleaned_data = clean_data(data, dim=2, vars_to_drop=['T', 'U:0', 'U:1', 'U:2'])

    embedding, mapper = embed_data(
        data=cleaned_data,
        algorithm=umap.UMAP,
        scale=True,
        **params
    )

    #umap_plot(mapper)

    #clusterer = cluster_embedding(
    #    embedding=embedding,
    #    algorithm=hdbscan.HDBSCAN,
    #    min_cluster_size=30
    #)

    plot_embedding(embedding=embedding, data=data, scale_points=True)

    #plot_cluster_membership(embedding=embedding, clusterer=clusterer,
    #                        save=True, figname=figname, figpath='../../../dimred_figs/figs_param_study/')

# %% Define path and read data

    path = '../../data/LES/2D/2D_212_35.csv'
    data = import_csv_data(path)

# %% Define variables for parametric study

    floats_array = np.logspace(1, 3, 20)
    range_n_neighbors = (int(np.round(x)) for x in floats_array)
    range_min_dist = [0.2]

    for n in range_n_neighbors:
        for d in range_min_dist:
            figname = f'n{n}_d{d*100}'.split('.', 1)[0]
            print(f'{figname}')
            run_UMAP(data, figname=figname, n_neighbors=n, min_dist=d)

# %%
