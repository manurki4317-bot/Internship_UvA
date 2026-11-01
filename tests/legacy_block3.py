"""Copia literal del código del notebook del bloque 3 (celdas 12-14, 19 y 42), solo como referencia para los tests."""
import numpy as np
import pandas as pd


def legacy_average(pd_list):
    columns = ['Joint Ent', 'Mut Info', 'Total Corr', 'Oinfo', 'Sinfo', 'nplets']
    new_pd = [df[columns] for df in pd_list]
    nplets = new_pd[0]['nplets']
    no_nplets = [df.drop('nplets', axis=1) for df in new_pd]
    for df in no_nplets:
        for column in df.columns:
            zscore_column = column + '_zscore'
            df[zscore_column] = (df[column] - df[column].mean()) / df[column].std()
            min_val = df[column].min()
            max_val = df[column].max()
            range_val = max_val - min_val
            normalized_column = column + '_normalized'
            df[normalized_column] = (df[column] - min_val) / range_val if range_val != 0 else 0
    average_df = pd.concat([df for df in no_nplets]).groupby(level=0).mean()
    average_df['nplets'] = nplets.to_list()
    return average_df


def legacy_find_significant_triplets(df_real, df_random, base_column, metric='zscore', lower_percentile=2.5, upper_percentile=97.5):
    metric_column = f"{base_column}_{metric}"
    real_column_values = df_real[metric_column].values
    random_column_values = df_random[metric_column].values
    lower_bound = np.percentile(random_column_values, lower_percentile)
    upper_bound = np.percentile(random_column_values, upper_percentile)
    is_significant = (real_column_values > upper_bound) | (real_column_values < lower_bound)
    significant_triplets = df_real[is_significant].copy()
    significant_triplets['Category'] = np.where(significant_triplets[metric_column] < upper_bound, 'Synergistic', 'Redundant')
    return significant_triplets
