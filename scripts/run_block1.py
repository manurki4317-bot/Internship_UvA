"""Bloque 1 por línea de comandos: series temporales de varios sujetos -> un CSV de tripletes por sujeto.

Ejemplo:
  python scripts/run_block1.py --inputs "HCP_Data/HCP_new_LR/HCP_new_LR/AAL_timeseries_*.txt" --outdir Results_LR --workers 8

- Cada archivo: texto separado por tabuladores, sin cabecera, tiempo x regiones.
- Reanudable: si el CSV de un sujeto ya existe, se salta (un sujeto tarda ~3 min con 92 regiones).
- Necesita infotopo_server.py en el PYTHONPATH (o en la misma carpeta desde la que se ejecuta).
"""
import argparse
import concurrent.futures as cf
import glob
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.getcwd())
import hoi  # noqa: E402


def _run(args):
    path, out = args
    logging.disable(logging.CRITICAL)            # infotopo escribe miles de líneas de progreso
    df = pd.read_csv(path, sep="\t", header=None)
    hoi.compute_hoi(df).to_csv(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="patrón glob entre comillas")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    jobs = []
    for f in sorted(glob.glob(a.inputs)):
        out = os.path.join(a.outdir, "HOI_" + os.path.splitext(os.path.basename(f))[0] + ".csv")
        if os.path.exists(out):
            print("ya existe, salto:", out)
        else:
            jobs.append((f, out))
    print(f"{len(jobs)} sujetos por procesar con {a.workers} procesos")
    with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for out in ex.map(_run, jobs):
            print("listo:", out)


if __name__ == "__main__":
    main()
