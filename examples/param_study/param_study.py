# Add to path so that dimred is visible to this script
import sys
sys.path.append('../../')

# Import local functions from dimred
from dimred.dimred import *

def run_case(data, algorithm):
    cleaned_data = clean_data(data, dim=2, vars_to_drop=['T'])

    embedding, mapper = embed_data(
        data=cleaned_data,
        algorithm=algorithm,
        scale=True,
    )

    clusterer = cluster_embedding(
        embedding=embedding,
        algorithm=hdbscan.HDBSCAN,
        min_cluster_size=60
    )

    plot_cluster_membership(embedding=embedding, clusterer=clusterer,
                            save=True, figname=str(algorithm)[-6:-2], figpath='../../../dimred_figs/figs_param_study/')


if __name__ == '__main__':
    path = '../../data/LES/2D/2D_212_35.csv'
    data = import_csv_data(path)

    algorithms = [TSNE, umap.UMAP]

    for algorithm in algorithms:
        run_case(data, algorithm)
