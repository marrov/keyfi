# Add to path so that dimred is visible to this script
import sys
sys.path.append('../../')

# Import local functions from dimred
from dimred.dimred import *


def run_UMAP(data, figname, **params):
    cleaned_data = clean_data(data, dim=2, vars_to_drop=[
                              'T'])

    embedding, mapper = embed_data(
        data=cleaned_data,
        algorithm=umap.UMAP,
        scale=True,
        **params
    )

    plot_embedding(embedding=embedding, data=data, scale_points=True, save=True,
                   figname=figname, figpath='../../../dimred_figs/figs_param_study/')

    # clusterer = cluster_embedding(
    #    embedding=embedding,
    #    algorithm=hdbscan.HDBSCAN,
    #    min_cluster_size=60
    # )

    # plot_cluster_membership(embedding=embedding, clusterer=clusterer,
    #                        save=True, figname=figname, figpath='../../../dimred_figs/figs_param_study/')


if __name__ == '__main__':
    path = '../../data/input/2D_212_35.csv'
    data = import_csv_data(path)

    floats_array = np.logspace(0.3, 2.7, 20)
    range_n_neighbors = [int(np.round(x)) for x in floats_array]
    range_min_dist = [0.2]

    for n in range_n_neighbors:
        for d in range_min_dist:
            figname = f'n{n}_d{d*100}'.split('.', 1)[0]
            run_UMAP(data, figname=figname, n_neighbors=n, min_dist=d)
