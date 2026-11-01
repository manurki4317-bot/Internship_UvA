"""Construye el visor interactivo (HTML autocontenido) con el paquete hoi.

  python viz/build_viewer.py --triplets Average_Data/average_triplets.csv \
      --null Average_Data/aggregated_centralities1000copies0005fraction.csv --out viz/hoi_hub_viewer.html

Precalcula, para información de interacción y correlación total, en redundancia y sinergia, el hipergrafo, sus
centralidades (EC, BC, DC, y z-scores) y los umbrales del modelo nulo (p = 0,10 / 0,05 / 0,01). El HTML carga
Plotly desde un CDN (necesita internet al abrirlo) y pesa ~1,3 MB.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import hoi  # noqa: E402
from hoi import atlas as A  # noqa: E402

PS = ["0.1", "0.05", "0.01"]
MEASURES = ["EC", "BC", "DC", "EC_z", "BC_z", "DC_z"]
CONFIGS = {"mi_red": ("Mut Info_normalized", "redundancy"), "mi_syn": ("Mut Info_normalized", "synergy"),
           "tc_red": ("Total Corr_zscore", "redundancy"), "tc_syn": ("Total Corr_zscore", "synergy")}
# Hubs por pares (matriz binarizada, z > 2) copiados de las salidas de Pairwise_Analysis.ipynb (CodeBlock5).
PAIRWISE = {"EC": [26, 60, 21, 65, 22, 64, 24, 25, 61, 63], "DC": [13, 22, 26, 15, 21, 60], "BC": [13, 16, 22]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triplets", required=True)
    ap.add_argument("--null", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "hoi_hub_viewer.html"))
    ap.add_argument("--fraction", type=float, default=0.005)
    ap.add_argument("--avg-corr", default=None, help="CSV 92x92 de |corr| promedio (run_block5.py): calcula los hubs por pares en vez de usar los copiados del notebook")
    ap.add_argument("--brain-obj", default=None, help="opcional: .obj original (si no, se usa la malla simplificada incluida)")
    a = ap.parse_args()

    atlas = A.load_atlas()
    V, F = A.load_brain(a.brain_obj)
    df = pd.read_csv(a.triplets, index_col=0)
    null = pd.read_csv(a.null, usecols=MEASURES)
    thr = {m: {p: float(np.percentile(null[m], 100 * (1 - float(p)))) for p in PS} for m in MEASURES}

    cfg = {}
    # tamaño por defecto (fracción, con modelo nulo) y tamaño 1000 (manuscrito v1: sin nulo, umbral manual de EC)
    for key, (col, mode), nt in [(k, v, None) for k, v in CONFIGS.items()] + [(k + "_1000", v, 1000) for k, v in CONFIGS.items()]:
        top = hoi.select_triplets(df, column=col, mode=mode, fraction=a.fraction, n_triplets=nt)
        cent = hoi.centralities(hoi.adjacency(top["nplets"]), top["nplets"])
        d = {"tri": [[x - 1 for x in t] for t in top["nplets"]]}          # los CSV usan índices 1-based
        d.update({m: [float("%.5g" % x) for x in cent[m]] for m in MEASURES})
        cfg[key] = d
        print(key, {m: int(hoi.exceeds(cent[m], thr[m]["0.05"]).sum()) for m in MEASURES} if nt is None else f"{len(top)} tripletes, EC>0.12: {int((cent['EC'] > 0.12).sum())}")

    V = V.astype(float)
    brain = dict(x=np.round(V[:, 0], 1).tolist(), y=np.round(V[:, 1], 1).tolist(), z=np.round(V[:, 2], 1).tolist(),
                 i=F[:, 0].tolist(), j=F[:, 1].tolist(), k=F[:, 2].tolist())
    pairwise = PAIRWISE
    if a.avg_corr:
        pairwise = hoi.pairwise_hub_indices(pd.read_csv(a.avg_corr, header=None).to_numpy(float))
    data = dict(pw=pairwise, cfg=cfg, thr=thr, pos=atlas["pos"].tolist(), names=atlas["names"], subn=atlas["subnet"],
                cols=atlas["colors"], brain=brain, n=len(cfg["mi_red"]["tri"]))
    html = open(os.path.join(HERE, "viewer_template.html"), encoding="utf-8").read()
    open(a.out, "w", encoding="utf-8").write(html.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    print("guardado:", a.out, f"({os.path.getsize(a.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
