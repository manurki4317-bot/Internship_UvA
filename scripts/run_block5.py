"""Bloque 5 por línea de comandos: series temporales -> matriz promedio de |correlación| y hubs por pares.

Ejemplo:
  python scripts/run_block5.py --series "HCP_Data/HCP_new_LR/HCP_new_LR/AAL_timeseries_*.txt" --outdir Average_Data

Escribe average_abs_correlation.csv (92x92) y pairwise_hubs.csv. Con --exclude-diagonal el percentil se calcula solo con los pares
fuera de la diagonal (el original incluye la diagonal, lo que sube el umbral; ver README_REFACTOR.md).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import hoi  # noqa: E402
from hoi import atlas as A  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True, help="patrón glob entre comillas")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--percentile", type=float, default=95)
    ap.add_argument("--exclude-diagonal", action="store_true")
    a = ap.parse_args()

    avg, n = hoi.average_abs_correlation(a.series, return_n=True)
    m = hoi.threshold_matrices(avg, a.percentile, include_diagonal=not a.exclude_diagonal)
    table = hoi.centrality_table(m["binarized"], m["weighted"])
    names = A.load_atlas()["names"]
    os.makedirs(a.outdir, exist_ok=True)
    pd.DataFrame(avg).to_csv(os.path.join(a.outdir, "average_abs_correlation.csv"), index=False, header=False)
    hubs = table.assign(region=names)[lambda t: t[[c for c in t.columns if c.endswith("_hub")]].any(axis=1)]
    hubs.to_csv(os.path.join(a.outdir, "pairwise_hubs.csv"))
    print(f"{n} sujetos; umbral |corr| > {m['critical']:.4f}; {int((m['binarized'].sum() - np.trace(m['binarized'])) // 2)} enlaces")
    for k, idx in hoi.pairwise_hubs(table).items():
        print(f"{k}: " + ", ".join(names[i] for i in idx))


if __name__ == "__main__":
    main()
