"""Figuras estáticas (matplotlib) para notebooks, informes y artículos. Las versiones interactivas están en viz/."""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D

from .atlas import NETWORKS, region_order

HUB, BASE = "#c2185b", "#5b78a8"
VIEWS = {"Axial": (0, 1), "Coronal": (0, 2), "Sagital": (1, 2)}   # ejes MNI (x, y, z) que se proyectan


def pair_counts(atlas, triplets, order=None):
    """Matriz regiones x regiones: nº de tripletes que contienen cada par (grado generalizado de la arista).
    `triplets`: índices 0-based. Devuelve (matriz ordenada por subred, orden usado)."""
    n = len(atlas["names"])
    order = order if order is not None else region_order(atlas)
    pos = {r: k for k, r in enumerate(order)}
    Z = np.zeros((n, n), int)
    for t in triplets:
        for a in range(3):
            for b in range(a + 1, 3):
                Z[pos[t[a]], pos[t[b]]] += 1
                Z[pos[t[b]], pos[t[a]]] += 1
    return Z, order


def plot_hub_projections(atlas, triplets, is_hub, brain=None, title=None, show_rest=True):
    """Tres proyecciones (axial, coronal, sagital) con los triángulos de cada hiperarista.
    Hubs en magenta, resto en azul tenue; nodos coloreados por subred y escalados por participación en hubs."""
    triplets, is_hub = np.asarray(triplets), np.asarray(is_hub, bool)
    pos = atlas["pos"]
    part = np.bincount(triplets[is_hub].ravel(), minlength=len(pos)) if is_hub.any() else np.zeros(len(pos), int)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, (name, (i, j)) in zip(axes, VIEWS.items()):
        if brain is not None:
            ax.scatter(brain[0][:, i], brain[0][:, j], s=1.5, c="#d6d9de", alpha=.35, lw=0, rasterized=True)
        if show_rest and (~is_hub).any():
            ax.add_collection(PolyCollection([pos[t][:, [i, j]] for t in triplets[~is_hub]], fc=BASE, ec=BASE, alpha=.07, lw=.3))
        if is_hub.any():
            ax.add_collection(PolyCollection([pos[t][:, [i, j]] for t in triplets[is_hub]], fc=HUB, ec=HUB, alpha=.22, lw=.5))
        ax.scatter(pos[:, i], pos[:, j], s=14 + 22 * np.sqrt(part), c=atlas["colors"], ec="k", lw=.4, zorder=3)
        ax.set_title(name); ax.set_aspect("equal"); ax.axis("off")
        ax.autoscale_view()
    handles = [Line2D([], [], marker="o", ls="", mfc=atlas["colors"][atlas["subnet"].index(k)], mec="k", label=v) for k, v in NETWORKS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=9)
    fig.suptitle(title or f"{int(is_hub.sum())} hubs de {len(triplets)} hiperaristas", y=.98)
    fig.tight_layout(rect=(0, .06, 1, .96))
    return fig


def plot_pair_matrix(atlas, triplets, title="Triángulos que contienen cada par de regiones", top=8):
    """Matriz 92x92 ordenada por subred, con las fronteras entre subredes y los pares más frecuentes anotados."""
    Z, order = pair_counts(atlas, triplets)
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    im = ax.imshow(Z, cmap="magma_r", interpolation="nearest")
    start, ticks, labels = 0, [], []
    for k, v in NETWORKS.items():
        n = sum(atlas["subnet"][r] == k for r in order)
        ticks.append(start + (n - 1) / 2); labels.append(v)
        start += n
        if start < len(order):
            ax.axhline(start - .5, c="w", lw=.8); ax.axvline(start - .5, c="w", lw=.8)
    ax.set_xticks(ticks, labels, rotation=40, ha="right"); ax.set_yticks(ticks, labels)
    fig.colorbar(im, ax=ax, shrink=.8, label="nº de triángulos")
    iu = np.triu_indices_from(Z, 1)
    best = np.argsort(Z[iu])[::-1][:top]
    txt = "\n".join(f"{atlas['names'][order[iu[0][b]]]} – {atlas['names'][order[iu[1][b]]]}: {Z[iu][b]}" for b in best if Z[iu][b])
    ax.set_title(title)
    fig.text(.02, .005, txt, fontsize=6.5, va="bottom", family="monospace", alpha=.8)
    fig.subplots_adjust(bottom=.2 + .012 * top)
    return fig


def plot_null_vs_real(null_vals, real_vals, threshold, label="EC (z-score)"):
    """Distribución de una centralidad en el modelo nulo frente a los hipergrafos reales, con el umbral."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(min(np.min(null_vals), np.min(real_vals)), max(np.max(null_vals), np.max(real_vals)), 70)
    ax.hist(null_vals, bins, density=True, alpha=.55, color="#8a94a6", label="modelo nulo (hipergrafos aleatorios)")
    ax.hist(real_vals, bins, density=True, alpha=.6, color=HUB, label="hipergrafo real")
    ax.axvline(threshold, c="k", ls="--", lw=1.2, label=f"umbral = {threshold:.3g}")
    ax.set_yscale("log"); ax.set_xlabel(label); ax.set_ylabel("densidad (escala log)"); ax.legend(frameon=False, fontsize=9)
    ax.set_title("Hubs: real frente a nulo")
    fig.tight_layout()
    return fig


def plot_pairwise_vs_highorder(atlas, pairwise_nodes, hub_triplets, title="Por pares frente a alto orden, por subred"):
    """% de hubs por pares y % de la participación en hubs de alto orden, agrupados por subred."""
    n = len(atlas["names"])
    cnt = np.bincount(np.asarray(hub_triplets).ravel(), minlength=n).astype(float)
    pw = np.zeros(n); pw[list(pairwise_nodes)] = 1
    keys = list(NETWORKS)
    a = [100 * pw[[r for r in range(n) if atlas["subnet"][r] == k]].sum() / pw.sum() for k in keys]
    h = [100 * cnt[[r for r in range(n) if atlas["subnet"][r] == k]].sum() / cnt.sum() for k in keys]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - .2, a, .4, color="#8a94a6", label="hubs por pares")
    ax.bar(x + .2, h, .4, color=[atlas["colors"][atlas["subnet"].index(k)] for k in keys], ec="k", lw=.4, label="participación en hubs de alto orden")
    ax.set_xticks(x, [NETWORKS[k] for k in keys], rotation=25, ha="right"); ax.set_ylabel("% del total")
    ax.legend(frameon=False, fontsize=9); ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_surrogate_check(real, surrogates, title="Surrogate check: which properties does each method preserve?"):
    """Compares real data and surrogates (dict name -> array time x regions) in three panels:
    lag-1 temporal autocorrelation, mean correlation between regions and distribution of the O-information of the triplets."""
    from .information import gaussian_copula_cov, soinfo_triplets
    sets = {"real data": np.asarray(real, float), **{k: np.asarray(v, float) for k, v in surrogates.items()}}
    ac = lambda m: np.mean([np.corrcoef(m[:-1, j], m[1:, j])[0, 1] for j in range(m.shape[1])])
    rr = lambda m: np.abs(np.corrcoef(m.T)[np.triu_indices(m.shape[1], 1)]).mean()
    colors = ["#222222", "#c2185b", "#5b78a8", "#2e8b57", "#e08a00"][: len(sets)]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for a, f, lab in ((ax[0], ac, "lag-1 autocorrelation"), (ax[1], rr, "mean |r| between regions")):
        a.bar(range(len(sets)), [f(m) for m in sets.values()], color=colors)
        a.set_xticks(range(len(sets)), list(sets), rotation=30, ha="right", fontsize=8); a.set_title(lab)
    allo = {k: soinfo_triplets(gaussian_copula_cov(m), len(m))[1] for k, m in sets.items()}
    lo, hi = min(o.min() for o in allo.values()), max(o.max() for o in allo.values())
    for (k, o), c in zip(allo.items(), colors):
        ax[2].hist(o, np.linspace(lo, hi, 90), alpha=.55, color=c, label=k, density=True)
    ax[2].set_yscale("log"); ax[2].set_xlabel("O-information of the triplets (>0 redundancy, <0 synergy)")
    ax[2].legend(fontsize=7, frameon=False); ax[2].set_title("O-information distribution")
    fig.suptitle(title); fig.tight_layout()
    return fig


def plot_real_vs_random(real, random, base_column, metric="zscore", ax=None):
    """Distribution of a metric in the real triplets compared with the random (surrogate) ones."""
    col = f"{base_column}_{metric}"
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.6))
    allv = np.concatenate([real[col].dropna(), random[col].dropna()])
    bins = np.linspace(allv.min(), allv.max(), 200)
    ax.hist(random[col].dropna(), bins, density=True, histtype="step", lw=2, color="red", label="phase-randomised")
    ax.hist(real[col].dropna(), bins, density=True, histtype="step", lw=2, color="#1f4e9c", label="real")
    ax.set_xlabel(col); ax.set_ylabel("density"); ax.legend(frameon=False)
    return ax.figure


def plot_adjacency(A, title="Adyacencia entre tripletes (comparten 2 regiones)", ax=None):
    """Matriz binaria de adyacencia del hipergrafo (sustituye a los clustermap de seaborn del bloque 4, que avisaban
    de que la matriz 'parece una matriz de distancias' y la reordenaban)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.asarray(A), cmap="Blues", interpolation="nearest")
    ax.set_title(f"{title}\n{int(np.asarray(A).sum() // 2)} enlaces entre {len(A)} tripletes", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    return ax.figure


def animate_hubs_3d(atlas, triplets, is_hub, brain, path, frames=72, fps=15, elev=20, show_rest=False, title=None):
    """GIF giratorio de los hubs sobre el cerebro (sustituye a las animaciones 'movie' de plotly del bloque 4 / README).
    Matplotlib 3D: más sencillo y menos interactivo que el visor HTML; para explorar usa viz/hoi_hub_viewer.html."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    triplets, is_hub = np.asarray(triplets), np.asarray(is_hub, bool)
    pos = atlas["pos"]
    fig = plt.figure(figsize=(6.4, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    V = brain[0][:: max(len(brain[0]) // 3500, 1)]
    ax.scatter(V[:, 0], V[:, 1], V[:, 2], s=1, c="#b9c0cb", alpha=.35, lw=0)
    if show_rest and (~is_hub).any():
        ax.add_collection3d(Poly3DCollection([pos[t] for t in triplets[~is_hub]], facecolors=BASE, edgecolors=BASE, alpha=.05, lw=.2))
    if is_hub.any():
        ax.add_collection3d(Poly3DCollection([pos[t] for t in triplets[is_hub]], facecolors=HUB, edgecolors=HUB, alpha=.18, lw=.4))
    part = np.bincount(triplets[is_hub].ravel(), minlength=len(pos)) if is_hub.any() else np.zeros(len(pos))
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=6 + 14 * np.sqrt(part), c=atlas["colors"], ec="k", lw=.3, depthshade=False)
    ax.set_box_aspect((np.ptp(V[:, 0]), np.ptp(V[:, 1]), np.ptp(V[:, 2])))
    ax.set_axis_off()
    ax.set_title(title or f"{int(is_hub.sum())} hubs", fontsize=10)
    fig.tight_layout()
    anim = FuncAnimation(fig, lambda k: ax.view_init(elev=elev, azim=360 * k / frames), frames=frames)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=85)
    plt.close(fig)
    return path


def plot_pairwise_matrices(weighted, binarized, critical=None):
    """Mapas de calor de las matrices ponderada y binarizada del análisis por pares (sustituyen a los sns.heatmap)."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    im = ax[0].imshow(np.asarray(weighted), cmap="coolwarm", interpolation="nearest")
    ax[0].set_title("Matriz ponderada" + (f" (|corr| > {critical:.3f})" if critical is not None else ""))
    fig.colorbar(im, ax=ax[0], shrink=.8)
    ax[1].imshow(np.asarray(binarized), cmap="Blues", interpolation="nearest")
    ax[1].set_title("Matriz binarizada")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    return fig


def plot_pairwise_centralities(table, z_threshold=2):
    """Histogramas de EC, DC y BC (red binarizada) con el corte de hub (z = z_threshold)."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, (c, col) in zip(ax, (("EC", "#1f4e9c"), ("DC", "#2e8b57"), ("BC", "#c2185b"))):
        v = table[c].to_numpy()
        a.hist(v, 20, color=col, alpha=.75)
        a.axvline(v.mean() + z_threshold * v.std(), c="k", ls="--", lw=1.2, label=f"z = {z_threshold}")
        a.set_title(f"{c}: {int(table[c + '_hub'].sum())} hubs"); a.set_xlabel(c); a.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig
