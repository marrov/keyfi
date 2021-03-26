# Add to path so that dimred is visible to this script
import sys
sys.path.append('../../')

# Import local functions from dimred
from dimred.dimred import *


def run_UMAP(data, figname, **params):
    cleaned_data = clean_data(data, dim=2, vars_to_drop=['U:0', 'U:1', 'U:2'])

    embedding, mapper = embed_data(
        data=cleaned_data,
        algorithm=umap.UMAP,
        scale=True,
        **params
    )

    plot_embedding(embedding=embedding, data=data, scale_points=True, save=True,
                   figname=figname, figpath='../../../dimred_figs/figs_param_study/')


if __name__ == '__main__':
    path_input = '../../data/input/2D_848_140.vtk'
    data, _ = import_vtk_data(path_input)

    floats_array = np.logspace(1, 4, 1)
    range_n_neighbors = [int(np.round(x)) for x in floats_array]
    range_min_dist = np.linspace(0.2, 0.6, 1)

    for n in range_n_neighbors:
        for d in range_min_dist:
            figname = f'n{n}_d{d*100}'.split('.', 1)[0]
            run_UMAP(data, figname=figname, n_neighbors=n, min_dist=d)
