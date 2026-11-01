"""Bloque 3 reorganizado: de los CSV por sujeto al promedio de grupo, y selección de tripletes significativos.

Sustituye al código de los notebooks (que repetía el mismo bucle para 'real' y 'aleatorio' y cargaba todos los sujetos
a la vez en memoria). Mismo resultado: las columnas, el orden y los valores de average_triplets*.csv no cambian.
"""
import glob
import warnings

import numpy as np
import pandas as pd

COLUMNS = ["Joint Ent", "Mut Info", "Total Corr", "Oinfo", "Sinfo"]


def normalize_subject(df, columns=COLUMNS):
    """Añade, por sujeto, el z-score (desviación típica con ddof=1, como pandas) y la normalización mín-máx de cada columna.
    Orden de salida idéntico al del notebook: primero las columnas originales, luego <col>_zscore, <col>_normalized por columna."""
    out = df[columns].copy()
    for c in columns:
        out[c + "_zscore"] = (out[c] - out[c].mean()) / out[c].std()
        rng = out[c].max() - out[c].min()
        out[c + "_normalized"] = (out[c] - out[c].min()) / rng if rng != 0 else 0
    return out


def find_subject_files(pattern):
    """Archivos que cumplen un patrón glob; si no hay ninguno, error que dice DÓNDE se ha buscado y qué hacer."""
    import os
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No hay archivos que cumplan el patrón:\n  {pattern}\n"
            f"Se ha buscado en:\n  {os.path.abspath(pattern)}\n"
            f"(carpeta de trabajo del notebook: {os.getcwd()})\n"
            "Comprueba la ruta, o usa MODE = 'averages' para partir de los promedios ya calculados (average_triplets_*.csv). "
            "Si los CSV del repo son punteros de Git LFS, descárgalos con `git lfs pull`.")
    return files


def average_subjects(files, columns=COLUMNS, return_n=False):
    """Promedio de grupo de los valores normalizados de cada sujeto. `files`: rutas de CSV (uno por sujeto, como las
    salidas del bloque 1) o patrón glob.

    Cambios respecto al notebook: lee y acumula de uno en uno (memoria constante), y COMPRUEBA que todos los sujetos
    listan los tripletes en el mismo orden (antes se asumía y se copiaba 'nplets' del primero)."""
    if isinstance(files, str):
        files = find_subject_files(files)
    if not files:
        raise FileNotFoundError("no hay archivos de sujetos")
    total = count = nplets = None
    for k, f in enumerate(files):
        raw = pd.read_csv(f, index_col=0)
        if "nplets" not in raw.columns or len(raw.columns) < len(columns) + 1:
            raise ValueError(f"{f}: no parece un CSV de tripletes (columnas: {list(raw.columns)}). "
                             "Si el archivo empieza por 'version https://git-lfs...' es un puntero de Git LFS: haz `git lfs pull`.")
        if nplets is None:
            nplets = raw["nplets"].reset_index(drop=True)
        elif not raw["nplets"].reset_index(drop=True).equals(nplets):
            raise ValueError(f"{f}: los tripletes no están en el mismo orden que en {files[0]}")
        norm = normalize_subject(raw, columns).reset_index(drop=True)
        vals = norm.to_numpy(float)
        if total is None:
            total, count, cols = np.zeros_like(vals), np.zeros_like(vals), list(norm.columns)
        ok = ~np.isnan(vals)
        total += np.where(ok, vals, 0)
        count += ok
    avg = pd.DataFrame(total / np.where(count == 0, np.nan, count), columns=cols)   # como groupby().mean(): ignora NaN
    avg["nplets"] = nplets.to_list()
    return (avg, len(files)) if return_n else avg


def find_significant_triplets(df_real, df_random, base_column, metric="zscore", lower_percentile=2.5, upper_percentile=97.5,
                              check_alignment=True):
    """Tripletes reales fuera del intervalo [p_inf, p_sup] de la distribución surrogate, con su categoría.

    Misma lógica y resultado que la función del notebook. Añadido: `check_alignment` verifica que real y aleatorio tienen
    los mismos tripletes en el mismo orden (la comparación es por distribución, pero un desorden indicaría un error).
    OJO con la etiqueta: 'Synergistic' = por debajo del límite inferior y 'Redundant' = por encima del superior. Eso solo tiene
    sentido para métricas con signo (Mut Info, Oinfo); para Total Corr o Joint Ent es simplemente 'bajo'/'alto'."""
    if metric not in ("zscore", "normalized"):
        raise ValueError("metric debe ser 'zscore' o 'normalized'")
    col = f"{base_column}_{metric}"
    if col not in df_real.columns or col not in df_random.columns:
        raise ValueError(f"la columna {col} no existe en alguno de los DataFrames")
    if check_alignment and "nplets" in df_real and "nplets" in df_random and \
            not df_real["nplets"].reset_index(drop=True).equals(df_random["nplets"].reset_index(drop=True)):
        warnings.warn("real y aleatorio no tienen los mismos tripletes en el mismo orden")
    lo, hi = np.percentile(df_random[col].values, [lower_percentile, upper_percentile])
    r = df_real[col].values
    sig = df_real[(r > hi) | (r < lo)].copy()
    sig["Category"] = np.where(sig[col] < hi, "Synergistic", "Redundant")
    return sig


def summarize_significance(df_real, df_random, columns=COLUMNS, metric="zscore", **kw):
    """Tabla resumen: nº y % de tripletes significativos (y por categoría) para cada métrica."""
    rows = []
    for c in columns:
        s = find_significant_triplets(df_real, df_random, c, metric, check_alignment=False, **kw)
        rows.append(dict(metrica=c, significativos=len(s), pct=100 * len(s) / len(df_real),
                         redundantes=int((s["Category"] == "Redundant").sum()), sinergicos=int((s["Category"] == "Synergistic").sum())))
    return pd.DataFrame(rows)
