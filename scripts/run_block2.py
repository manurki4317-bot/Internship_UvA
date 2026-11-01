"""Bloque 2 por línea de comandos: genera un surrogate por sujeto (aleatorización de fase).

Ejemplo:
  python scripts/run_block2.py --inputs "HCP_Data/HCP_new_LR/HCP_new_LR/AAL_timeseries_*.txt" --outdir PR_LR --mode independent --seed 42 --workers 8

- Entrada: texto separado por tabuladores, sin cabecera, tiempo x regiones (como en el bloque 1).
- --mode: independent (recomendado: coincide con "aleatorizar cada serie temporal"), shared, o legacy (función original).
- Reproducible: cada sujeto usa una semilla derivada de --seed y del orden alfabético de los archivos.
- Reanudable: salta los sujetos cuyo CSV de salida ya existe.
"""
import argparse
import concurrent.futures as cf
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import hoi  # noqa: E402


def _run(job):
    path, out, mode, seed = job
    df = pd.read_csv(path, sep="\t", header=None)
    hoi.make_surrogate(df, mode, seed).to_csv(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--mode", default="independent", choices=list(hoi.surrogates.MODES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()

    files = sorted(glob.glob(a.inputs))
    seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(a.seed).spawn(len(files))]
    os.makedirs(a.outdir, exist_ok=True)
    jobs = []
    for f, s in zip(files, seeds):
        out = os.path.join(a.outdir, "PR_" + os.path.splitext(os.path.basename(f))[0] + ".csv")
        if os.path.exists(out):
            print("ya existe, salto:", out)
        else:
            jobs.append((f, out, a.mode, s))
    print(f"{len(jobs)} sujetos, modo {a.mode}, {a.workers} procesos")
    with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for out in ex.map(_run, jobs):
            print("listo:", out)


if __name__ == "__main__":
    main()
