"""Modelo nulo: centralidades de hipergrafos formados por tripletes elegidos al azar."""
import concurrent.futures as cf
import numpy as np
import pandas as pd
from .hypergraph import select_triplets, adjacency, centralities

_DF = None
_FRACTION = None
_N = None


def _init(df, fraction, n_triplets):
    global _DF, _FRACTION, _N
    _DF, _FRACTION, _N = df, fraction, n_triplets


def _one(seed):
    top = select_triplets(_DF, mode="random", fraction=_FRACTION, random_state=seed, n_triplets=_N)
    return centralities(adjacency(top["nplets"]), top["nplets"])


def null_distribution(df, n_copies=1000, fraction=0.005, workers=None, seed=0, n_triplets=None):
    """Concatena las centralidades de `n_copies` hipergrafos aleatorios (reproducible con `seed`)."""
    seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(n_copies)]
    with cf.ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(df, fraction, n_triplets)) as ex:
        results = list(ex.map(_one, seeds, chunksize=8))
    return pd.concat(results, ignore_index=True)
