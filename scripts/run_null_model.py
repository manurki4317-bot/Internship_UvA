"""Sustituye a surrogate_analysis.py: genera el modelo nulo y escribe el CSV de centralidades.

Ejemplo:
  python scripts/run_null_model.py --triplets Average_Data/average_triplets.csv --copies 1000 --workers 10
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import hoi  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triplets", required=True, help="CSV con las columnas de valores y 'nplets'")
    ap.add_argument("--copies", type=int, default=1000)
    ap.add_argument("--fraction", type=float, default=0.005)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    df = pd.read_csv(a.triplets, index_col=0)
    null = hoi.null_distribution(df, a.copies, a.fraction, a.workers, a.seed)
    out = a.out or f"aggregated_centralities{a.copies}copies{str(a.fraction).replace('.', '')}fraction.csv"
    null.to_csv(out, index=False)
    print("Guardado:", out)
    for p in (0.10, 0.05, 0.01):
        thr = hoi.hub_thresholds(null, p)
        print(f"p={p}: " + ", ".join(f"{k}>{v:.4g}" for k, v in thr.items()))


if __name__ == "__main__":
    main()
