"""Bloque 5 reorganizado: análisis por pares (correlación) para comparar con el de alto orden.

Pasos del notebook original, ahora como funciones: promedio de |correlación| entre sujetos -> umbral por percentil ->
matrices binarizada y ponderada -> centralidades (EC, DC, BC) y hubs (z-score > 2).
"""
import glob

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import zscore

from .hypergraph import eigenvector_centrality


def abs_correlation(ts):
    """|correlación de Pearson| entre regiones de una serie (tiempo x regiones). Sin NaN: se descartan las filas con NaN."""
    ts = pd.DataFrame(ts).dropna()
    return np.abs(np.corrcoef(ts.to_numpy(float), rowvar=False))


def average_abs_correlation(files, return_n=False):
    """Media entre sujetos de |corr|. `files`: rutas de series (tabuladores, sin cabecera, tiempo x regiones) o patrón glob.
    Lee de uno en uno (memoria constante). El original recortaba siempre la última fila; aquí se descartan las filas con NaN."""
    if isinstance(files, str):
        files = sorted(glob.glob(files))
    if not files:
        raise FileNotFoundError("no hay series temporales: comprueba el patrón de archivos")
    total = None
    for f in files:
        c = abs_correlation(pd.read_csv(f, sep="\t", header=None))
        total = c if total is None else total + c
    avg = total / len(files)
    return (avg, len(files)) if return_n else avg


def threshold_matrices(avg, percentile=95, include_diagonal=True):
    """Umbral por percentil de la distribución de valores de la matriz promedio, y matrices binarizada y ponderada.

    include_diagonal=True reproduce el notebook original: el percentil se calcula con TODA la matriz aplanada (incluye la diagonal,
    que vale 1, y cada par dos veces) y la diagonal queda como autoenlaces. include_diagonal=False usa solo los pares fuera de la
    diagonal y no deja autoenlaces. Devuelve dict(critical, binarized, weighted)."""
    avg = np.asarray(avg, float)
    if include_diagonal:
        values = avg.flatten()
    else:
        values = avg[~np.eye(len(avg), dtype=bool)]
    crit = float(np.percentile(values, percentile))
    sig = avg > crit
    if not include_diagonal:
        np.fill_diagonal(sig, False)
    return dict(critical=crit, binarized=sig.astype(int), weighted=np.where(sig, avg, 0.0))


def centrality_table(binarized, weighted=None, z_threshold=2):
    """EC, DC, BC de la red binarizada (y EC_w, BC_w de la ponderada) con sus z-scores (ddof=0, como el original) y marca de hub.

    - EC: vector propio dominante (hoi.hypergraph.eigenvector_centrality), igual que nx.eigenvector_centrality hasta ~1e-6 pero sin
      el error de convergencia de la iteración de potencias.
    - OJO BC ponderada: networkx interpreta `weight` como DISTANCIA, y aquí el peso es la fuerza de la correlación (mayor = más
      cerca). El original lo hace así y se mantiene por fidelidad; para una BC con sentido habría que usar distancias (1/peso)."""
    Gb = nx.from_numpy_array(np.asarray(binarized, float))
    out = pd.DataFrame({"EC": eigenvector_centrality(binarized), "DC": list(nx.degree_centrality(Gb).values()),
                        "BC": list(nx.betweenness_centrality(Gb).values())})
    if weighted is not None:
        Gw = nx.from_numpy_array(np.asarray(weighted, float))
        out["EC_w"] = eigenvector_centrality(weighted)
        out["BC_w"] = list(nx.betweenness_centrality(Gw, weight="weight").values())
    for c in list(out.columns):
        out[c + "_z"] = zscore(out[c])
    for c in [c for c in out.columns if not c.endswith("_z")]:
        out[c + "_hub"] = out[c + "_z"] > z_threshold
    return out


def pairwise_hubs(table, binarized_only=True):
    """Índices (0-based) de los hubs por pares de cada centralidad, ordenados de mayor a menor z-score."""
    cols = ["EC", "DC", "BC"] if binarized_only else [c for c in table.columns if c.endswith("_hub")]
    cols = [c.replace("_hub", "") for c in cols]
    return {c: table.index[table[c + "_hub"]].tolist() if False else
            table[table[c + "_hub"]].sort_values(c + "_z", ascending=False).index.tolist() for c in cols}


def pairwise_hub_indices(avg, percentile=95, include_diagonal=True):
    """Atajo: de la matriz promedio de |corr| a {'EC': [...], 'DC': [...], 'BC': [...]} (red binarizada, z > 2)."""
    m = threshold_matrices(avg, percentile, include_diagonal)
    return pairwise_hubs(centrality_table(m["binarized"], m["weighted"]))
