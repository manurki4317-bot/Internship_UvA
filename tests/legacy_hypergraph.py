"""Copia literal de create_hypergraph y HO_cent_df de surrogate_analysis.py (solo referencia para los tests)."""
import ast
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import zscore


def create_hypergraph(dataframe, fraction=0.005, sort_column='Mut Info_normalized', mode='redundancy', random_selection=False):
    if random_selection:
        shuffled_df = dataframe.sample(frac=1).reset_index(drop=True)
    else:
        if mode == 'redundancy':
            shuffled_df = dataframe.sort_values(sort_column, ascending=False).copy()
        elif mode == 'synergy':
            shuffled_df = dataframe.sort_values(sort_column, ascending=True).copy()
        else:
            raise ValueError("Mode must be either 'redundancy' or 'synergy'")
    num_rows = int(len(shuffled_df) * fraction)
    top_df = shuffled_df.head(num_rows).copy()
    top_df['nplets'] = top_df['nplets'].apply(ast.literal_eval)
    n = len(top_df['nplets'])
    H = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if len(set(top_df['nplets'].iloc[i]).intersection(set(top_df['nplets'].iloc[j]))) == 2:
                H[i, j] = 1
                H[j, i] = 1
    return top_df, H


def HO_cent_df(hypergraph, dataframe):
    HG = nx.from_numpy_array(hypergraph)
    EC = list(nx.eigenvector_centrality_numpy(HG).values())
    BC = list(nx.betweenness_centrality(HG).values())
    DC = list(nx.degree_centrality(HG).values())
    HO_cent = pd.DataFrame(list(zip(EC, BC, DC)), columns=['EC', 'BC', 'DC'])
    HO_cent['EC_z'] = zscore(HO_cent['EC'])
    HO_cent['BC_z'] = zscore(HO_cent['BC'])
    HO_cent['DC_z'] = zscore(HO_cent['DC'])
    HO_cent['nplets'] = dataframe['nplets'].to_list()
    return HO_cent
