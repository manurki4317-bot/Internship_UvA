"""Bloque 1 reorganizado: de una serie temporal (tiempo x regiones) a la tabla de tripletes.

Sustituye a run_all_HOI / run_High_Order / info_topo de HOI_connectivity.py. Devuelve las mismas
columnas y en el mismo orden: Oinfo, Sinfo, Joint Ent, Mut Info, Total Corr, nplets.

- O-info y S-info: hoi.information (vectorizado, sin infotopo).
- Entropía conjunta, información de interacción ('Mut Info') y correlación total: librería infotopo
  (Baudot), que usa probabilidades discretizadas (20 valores); se conserva tal cual.
"""
import importlib

import numpy as np
import pandas as pd

from .information import gaussian_copula_cov, soinfo_triplets


def clean_timeseries(df):
    """Quita las filas con NaN (el legado recortaba siempre la última fila con df.iloc[:-1])."""
    return df.dropna()


def _load_infotopo():
    try:
        return importlib.import_module("infotopo_server")
    except ImportError as e:  # pragma: no cover
        raise ImportError("Necesito infotopo_server.py en el PYTHONPATH (librería de P. Baudot).") from e


def entropy_metrics(ts, nb_of_values=20, infotopo_module=None):
    """Joint Ent, Mut Info y Total Corr de todos los tripletes, con infotopo.

    ts: array (T, N). Devuelve un DataFrame con esas tres columnas y 'nplets' (índices de infotopo).
    """
    it = infotopo_module or _load_infotopo()
    N = ts.shape[1]
    model = it.infotopo(dimension_max=3, dim_to_rank=2, number_of_max_val=5, dimension_tot=N,
                        nb_of_values=nb_of_values, forward_computation_mode=True)
    joint = model.simplicial_entropies_decomposition(ts)
    mut = model.simplicial_infomut_decomposition(joint)
    tc = model.total_correlation_simplicial_lanscape(joint)
    keys = list(joint.keys())
    if list(mut.keys()) != keys or list(tc.keys()) != keys:
        raise RuntimeError("infotopo devolvió los simplices en distinto orden; no se pueden alinear las columnas.")
    mask = [len(k) == 3 for k in keys]
    sel = lambda d: [v for v, m in zip(d.values(), mask) if m]
    return pd.DataFrame({"Joint Ent": sel(joint), "Mut Info": sel(mut), "Total Corr": sel(tc),
                         "nplets": [list(k) for k, m in zip(keys, mask) if m]})


def compute_hoi(df, nb_of_values=20, infotopo_module=None):
    """Tabla de tripletes para un sujeto. df: DataFrame tiempo x regiones."""
    ts = clean_timeseries(df).to_numpy(dtype=float)
    T = len(ts)
    ent = entropy_metrics(ts, nb_of_values, infotopo_module)
    nplets0, oinfo, sinfo = soinfo_triplets(gaussian_copula_cov(ts), T, n=3)
    # Comprobación que el código original no hacía: ambos cálculos deben listar los mismos tripletes.
    if not (np.asarray(ent["nplets"].tolist()) - 1 == nplets0).all():
        raise RuntimeError("Los tripletes de infotopo y de O/S-info no están en el mismo orden.")
    out = pd.DataFrame({"Oinfo": oinfo, "Sinfo": sinfo})
    return pd.concat([out, ent], axis=1)
