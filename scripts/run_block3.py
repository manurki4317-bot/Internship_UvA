"""Bloque 3 por línea de comandos: CSV por sujeto (reales y surrogate) -> average_triplets.csv / average_triplets_random.csv.

Ejemplo:
  python scripts/run_block3.py --real "Data/Real/Results_RL/*.csv" --random "Data/Surrogate/*.csv" --outdir Average_Data

Usa los mismos CSV por sujeto que produce el bloque 1 (sobre las series reales y sobre las surrogate del bloque 2).
Si los archivos son punteros de Git LFS, avisa de que hay que hacer `git lfs pull`.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import hoi  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True, help="patrón glob entre comillas")
    ap.add_argument("--random", required=True, help="patrón glob entre comillas")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--suffix", default="", help="p. ej. _RL para escribir average_triplets_RL.csv (recomendado: nombrar por adquisición)")
    a = ap.parse_args()

    real, n_real = hoi.average_subjects(hoi.find_subject_files(a.real), return_n=True)
    rand, n_rand = hoi.average_subjects(hoi.find_subject_files(a.random), return_n=True)
    if n_real != n_rand:
        print(f"AVISO: {n_real} sujetos reales frente a {n_rand} surrogate; las distribuciones de promedios no son comparables.")
    if real["nplets"].tolist() != rand["nplets"].tolist():
        print("AVISO: real y surrogate no tienen los mismos tripletes en el mismo orden.")
    os.makedirs(a.outdir, exist_ok=True)
    real.to_csv(os.path.join(a.outdir, f"average_triplets{a.suffix}.csv"))
    rand.to_csv(os.path.join(a.outdir, f"average_triplets_random{a.suffix}.csv"))
    print(hoi.summarize_significance(real, rand).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
