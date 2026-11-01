"""Copia literal del código del notebook del bloque 5 (celdas 4-5, 8 y 15), solo como referencia para los tests."""
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import zscore


def legacy_average(series_list):
    df_list = []
    for df in series_list:
        df = df.iloc[:-1, :]
        df_list.append(np.abs(df.corr()))
    return pd.concat(df_list).groupby(level=0).mean()


def legacy_matrices(average_df):
    empirical_distribution = average_df.values.flatten()
    upper_critical_value = np.percentile(empirical_distribution, 95)
    significant_matrix = (average_df > upper_critical_value)
    weighted_matrix = average_df.copy()
    weighted_matrix[~significant_matrix] = 0
    binarized_matrix = significant_matrix.astype(int)
    return upper_critical_value, binarized_matrix, weighted_matrix


def legacy_hubs(binarized_matrix, weighted_matrix, z_threshold=2):
    def compute_centrality(G, centrality_func, name, use_weights=False):
        centrality = centrality_func(G, weight='weight') if use_weights else centrality_func(G)
        df = pd.DataFrame(list(centrality.items()), columns=['Node', name])
        df['ZScore'] = zscore(df[name])
        return df[df['ZScore'] > z_threshold].sort_values(by='ZScore', ascending=False)
    Gb = nx.from_pandas_adjacency(binarized_matrix)
    Gw = nx.from_pandas_adjacency(weighted_matrix)
    return {
        'EC': compute_centrality(Gb, nx.eigenvector_centrality, 'EC'),
        'DC': compute_centrality(Gb, nx.degree_centrality, 'DC'),
        'BC': compute_centrality(Gb, nx.betweenness_centrality, 'BC'),
        'EC_w': compute_centrality(Gw, nx.eigenvector_centrality, 'EC', use_weights=True),
        'BC_w': compute_centrality(Gw, nx.betweenness_centrality, 'BC', use_weights=True)}
