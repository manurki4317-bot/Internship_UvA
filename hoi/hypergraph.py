"""Construcción del hipergrafo de tripletes y sus centralidades.

Sustituye a create_hypergraph / HO_cent_df, que estaban duplicadas en el notebook
del bloque 4 y en surrogate_analysis.py. Mismo resultado, sin doble bucle en Python.
"""
import ast
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import zscore

MEASURES = ("EC", "BC", "DC")
TIE_RTOL = 1e-9   # tolerancia RELATIVA: un valor que solo supera el umbral por ruido de coma flotante (p. ej. DC, discreto) no cuenta


def select_triplets(df, column="Mut Info_normalized", mode="redundancy", fraction=0.005, random_state=None, n_triplets=None):
    """Devuelve el `fraction` superior de tripletes (con 'nplets' ya convertido a tupla).

    mode: 'redundancy' (mayores valores), 'synergy' (menores) o 'random' (modelo nulo).
    n_triplets: si se da, número fijo de tripletes (anula `fraction`). El manuscrito v1 usa n_triplets=1000.
    """
    n = int(n_triplets) if n_triplets is not None else int(len(df) * fraction)
    if mode == "random":
        top = df.sample(n=n, random_state=random_state)
    elif mode in ("redundancy", "synergy"):
        top = df.sort_values(column, ascending=(mode == "synergy")).head(n)
    else:
        raise ValueError("mode debe ser 'redundancy', 'synergy' o 'random'")
    top = top.copy()
    top["nplets"] = top["nplets"].apply(lambda s: tuple(ast.literal_eval(s)) if isinstance(s, str) else tuple(s))
    return top


def incidence(triplets):
    """Matriz de incidencia tripletes x nodos (0/1)."""
    t = np.asarray(list(triplets))
    _, inv = np.unique(t, return_inverse=True)
    inv = inv.reshape(t.shape)
    M = np.zeros((len(t), inv.max() + 1), dtype=np.int32)
    M[np.arange(len(t))[:, None], inv] = 1
    return M


def adjacency(triplets):
    """Dos tripletes son vecinos si comparten exactamente 2 nodos (producto de incidencias)."""
    M = incidence(triplets)
    return ((M @ M.T) == 2).astype(int)


def eigenvector_centrality(A):
    """Vector propio dominante de la adyacencia (norma 1, signo positivo).

    networkx >= 3 lanza AmbiguousSolution en grafos desconectados, como los hipergrafos
    aleatorios del modelo nulo; networkx 2.4 (usado originalmente) devolvía este mismo vector.
    """
    w, v = np.linalg.eigh(np.asarray(A, dtype=float))
    vec = v[:, -1]
    return vec / (np.sign(vec.sum() or 1.0) * np.linalg.norm(vec))


def centralities(A, nplets=None):
    """EC, BC y DC (y sus z-scores) de cada triplete a partir de la adyacencia."""
    G = nx.from_numpy_array(A)
    out = pd.DataFrame({
        "EC": eigenvector_centrality(A),
        "BC": list(nx.betweenness_centrality(G).values()),
        "DC": list(nx.degree_centrality(G).values()),
    })
    for m in MEASURES:
        out[m + "_z"] = zscore(out[m])
    if nplets is not None:
        out["nplets"] = list(nplets)
    return out


def hub_thresholds(null_df, p=0.05, measures=None):
    """Umbral = percentil (1-p) de la distribución nula, para cada medida."""
    cols = measures or [c for c in null_df.columns if c.split("_")[0] in MEASURES]
    return {c: float(np.percentile(null_df[c], 100 * (1 - p))) for c in cols}


def exceeds(values, threshold):
    """values > threshold, ignorando los empates que solo difieren por ruido de coma flotante."""
    values = np.asarray(values)
    return (values > threshold) & ~np.isclose(values, threshold, rtol=TIE_RTOL, atol=0)


def find_hubs(cent_df, thresholds, measure="EC", use_z=True):
    """Filas de `cent_df` cuya centralidad supera el umbral del modelo nulo."""
    col = measure + ("_z" if use_z else "")
    return cent_df[exceeds(cent_df[col].to_numpy(), thresholds[col])]
