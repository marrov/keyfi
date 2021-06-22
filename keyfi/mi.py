import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_regression


def get_mi_scores(X, y, scale: bool = False):
    mi_scores = mutual_info_regression(X, y)
    if scale:
        mi_scores /= np.max(mi_scores)
    return mi_scores


def fix_yticks(labels):
    new_yticks = []
    for label in labels:
        if '_' in label.get_text():
            # Fix species' names
            lst = label.get_text().split('_')
            lst.insert(1, '_{')
            lst.append('}')
            new_yticks.append(''.join(lst))
        elif ':' in label.get_text():
            # Fix velocity componenets' names
            lst = label.get_text().split(':')
            if lst[-1] == '0':
                lst[-1] = '_x'
            elif lst[-1] == '1':
                lst[-1] = '_y'
            elif lst[-1] == '2':
                lst[-1] = '_z'
            new_yticks.append(''.join(lst))
        else:
            new_yticks.append(label.get_text())

    new_yticks = ['$\mathrm{' + item + '}$' for item in new_yticks]
    return new_yticks


def plot_cluster_mi_scores(cluster_mi_scores):

    fig, ax = plt.subplots(figsize=[7, 7])
    palette = sns.color_palette('husl', 2)
    ax = sns.barplot(x='Mutual Information scores', y='Variables',
                     hue='Synthetic variables', data=cluster_mi_scores, palette=palette)
    ax.set_xlabel(xlabel='Mutual Information (MI) scores', fontsize=17)
    ax.set_ylabel(ylabel='Original variables', fontsize=17)
    locs, labels = plt.yticks()
    new_yticks = fix_yticks(labels=labels)
    plt.yticks(locs, new_yticks, fontsize=17)
    plt.xticks(fontsize=17)
    plt.legend(fontsize=17)
    plt.show()


def get_cluster_mi_scores(data, clusterer, embedding, cluster_num: int = 0, scale: bool = False, flag_print: bool = True, flag_plot: bool = True):
    clustered_data = data.copy()
    clustered_data['clusters'] = clusterer.labels_
    clustered_data[['Var_X', 'Var_Y']] = embedding

    cluster_target = clustered_data[(
        clustered_data['clusters'] == cluster_num)].copy()
    cluster_target.drop(columns='clusters', inplace=True)

    X = cluster_target.drop(columns=['Var_X', 'Var_Y'])
    y = cluster_target[['Var_X', 'Var_Y']]



    df_MI = pd.DataFrame(
        {
            'Variables': X.columns,
            'UMAP x-axis': get_mi_scores(X, y['Var_X'], scale=scale),
            'UMAP y-axis': get_mi_scores(X, y['Var_Y'], scale=scale)
        })

    cluster_mi_scores = df_MI.melt(id_vars=['Variables'], value_vars=['UMAP x-axis', 'UMAP y-axis'],
                                   var_name='Synthetic variables', value_name='Mutual Information scores')

    cluster_mi_scores.sort_values(
        'Mutual Information scores', ascending=False, inplace=True, ignore_index=True)

    if flag_print:
        print(f'Mutual Information scores for cluster {cluster_num}: \n')
        print(cluster_mi_scores)
        print('\n')

    if flag_plot:
        plot_cluster_mi_scores(cluster_mi_scores)

    return cluster_mi_scores
