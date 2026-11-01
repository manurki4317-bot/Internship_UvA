"""Genera las figuras estáticas (PNG) a partir de los CSV del proyecto.

  python viz/make_figures.py --triplets Average_Data/average_triplets.csv \
      --null Average_Data/aggregated_centralities1000copies0005fraction.csv --out figures
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import hoi  # noqa: E402
from hoi import atlas as A, plots  # noqa: E402

# Hubs por pares (matriz binarizada, z > 2) copiados de las salidas de Pairwise_Analysis.ipynb (CodeBlock5).
PAIRWISE_EC = [26, 60, 21, 65, 22, 64, 24, 25, 61, 63]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triplets", required=True)
    ap.add_argument("--null", required=True)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--p", type=float, default=0.05)
    ap.add_argument("--column", default="Mut Info_normalized")
    ap.add_argument("--mode", default="redundancy", choices=["redundancy", "synergy"])
    ap.add_argument("--brain-obj", default=None)
    ap.add_argument("--avg-corr", default=None, help="CSV 92x92 de |corr| promedio: calcula los hubs por pares (si no, usa los copiados del notebook)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    atlas, brain = A.load_atlas(), A.load_brain(a.brain_obj)
    df = pd.read_csv(a.triplets, index_col=0)
    null = pd.read_csv(a.null)
    top = hoi.select_triplets(df, column=a.column, mode=a.mode)
    cent = hoi.centralities(hoi.adjacency(top["nplets"]), top["nplets"])
    thr = hoi.hub_thresholds(null, a.p)
    is_hub = (cent["EC_z"] > thr["EC_z"]).to_numpy()
    tri = np.array(top["nplets"].tolist()) - 1            # los CSV usan índices 1-based
    tag = f"{a.column.split('_')[0].replace(' ', '')}_{a.mode}"

    pairwise_ec = hoi.pairwise_hub_indices(pd.read_csv(a.avg_corr, header=None).to_numpy(float))["EC"] if a.avg_corr else PAIRWISE_EC
    figs = {
        f"hubs_proyecciones_{tag}": plots.plot_hub_projections(atlas, tri, is_hub, brain, f"{int(is_hub.sum())} hubs (EC, z-score, p={a.p}) de {len(tri)} hiperaristas · {a.column} · {a.mode}"),
        f"matriz_pares_{tag}": plots.plot_pair_matrix(atlas, tri[is_hub], f"Triángulos hub que contienen cada par de regiones · {tag}"),
        f"nulo_vs_real_{tag}": plots.plot_null_vs_real(null["EC_z"], cent["EC_z"], thr["EC_z"]),
        f"pares_vs_altoorden_{tag}": plots.plot_pairwise_vs_highorder(atlas, pairwise_ec, tri[is_hub]),
    }
    for name, fig in figs.items():
        fig.savefig(os.path.join(a.out, name + ".png"), dpi=150)
        print("guardado:", os.path.join(a.out, name + ".png"))


if __name__ == "__main__":
    main()
