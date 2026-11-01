"""Atlas AAL92 (nombres, posiciones MNI, subredes y colores) y malla del cerebro."""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ATLAS_DIR = os.path.join(HERE, "..", "atlas", "AAL92")
BRAIN_NPZ = os.path.join(HERE, "..", "atlas", "brain_simplified.npz")
NETWORKS = {"VIS": "Visual", "SM": "Somatomotora", "DMN": "Modo por defecto", "FP": "Frontoparietal",
            "VA": "Atención ventral", "LIM": "Límbica", "SC": "Subcortical"}   # orden de dibujo


def load_atlas(annotation_dir=ATLAS_DIR, prefix="AAL92"):
    """Devuelve dict(names, pos (92x3), subnet, colors). Limpia los tabuladores de los nombres originales."""
    def rd(f):
        with open(os.path.join(annotation_dir, f"{prefix}_{f}.txt")) as fh:
            return [l.strip() for l in fh if l.strip()]
    names = [l.replace("\t", "").replace(" ", "_") for l in rd("region_names_full")]
    pos = np.loadtxt(os.path.join(annotation_dir, f"{prefix}_positions.txt"))
    subnet, colors = rd("subnet_ordernames"), rd("subnet_order_colors")
    assert len(names) == len(pos) == len(subnet) == len(colors), "los archivos del atlas no tienen el mismo nº de regiones"
    return dict(names=names, pos=pos, subnet=subnet, colors=colors)


def region_order(atlas):
    """Índices de regiones agrupados por subred (en el orden de NETWORKS) y luego por índice."""
    keys = list(NETWORKS)
    return sorted(range(len(atlas["names"])), key=lambda r: (keys.index(atlas["subnet"][r]), r))


def simplify_brain_obj(path, grid=3.0):
    """Lee un .obj y reduce vértices agrupándolos en una rejilla de `grid` mm. Devuelve (vértices, caras)."""
    V, F = [], []
    for l in open(path):
        if l.startswith("v "):
            V.append([float(x) for x in l.split()[1:4]])
        elif l.startswith("f "):
            F.append([int(t.split("/")[0]) - 1 for t in l.split()[1:4]])
    V, F = np.array(V), np.array(F)
    _, inv = np.unique(np.round(V / grid).astype(int), axis=0, return_inverse=True)
    inv = inv.ravel()
    cen = np.zeros((inv.max() + 1, 3))
    np.add.at(cen, inv, V)
    cen /= np.bincount(inv)[:, None]
    F2 = inv[F]
    F2 = F2[(F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2])]
    return cen, F2


def load_brain(obj_path=None):
    """Malla simplificada del cerebro: desde un .obj (si se da) o desde atlas/brain_simplified.npz."""
    if obj_path:
        return simplify_brain_obj(obj_path)
    z = np.load(BRAIN_NPZ)
    return z["V"], z["F"]
