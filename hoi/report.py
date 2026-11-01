"""Tablas de resultados: qué regiones forman los hubs y cuánto participan."""
import numpy as np
import pandas as pd

from .atlas import NETWORKS


def hub_table(atlas, cent_df, is_hub, measure="EC_z", triplets=None):
    """Una fila por hub (tripleta): regiones, subredes y centralidades, ordenadas de mayor a menor `measure`.
    `triplets`: índices 0-based; si no se da, se leen de cent_df['nplets'] (1-based, como en los CSV)."""
    tri = np.asarray(triplets if triplets is not None else [list(t) for t in cent_df["nplets"]])
    if triplets is None:
        tri = tri - 1
    is_hub = np.asarray(is_hub, bool)
    rows = []
    for t, (_, r) in zip(tri[is_hub], cent_df[is_hub].iterrows()):
        rows.append({"region_1": atlas["names"][t[0]], "region_2": atlas["names"][t[1]], "region_3": atlas["names"][t[2]],
                     "subredes": "/".join(sorted({atlas["subnet"][i] for i in t})),
                     "EC": r["EC"], "BC": r["BC"], "DC": r["DC"], "EC_z": r["EC_z"], "BC_z": r["BC_z"], "DC_z": r["DC_z"]})
    return pd.DataFrame(rows).sort_values(measure, ascending=False).reset_index(drop=True)


def region_participation(atlas, hub_triplets):
    """Nº de hubs en los que participa cada región, y su subred. `hub_triplets`: índices 0-based."""
    cnt = np.bincount(np.asarray(hub_triplets).ravel(), minlength=len(atlas["names"]))
    df = pd.DataFrame({"region": atlas["names"], "subred": atlas["subnet"], "hubs": cnt})
    return df.sort_values("hubs", ascending=False).reset_index(drop=True)


def subnetwork_share(atlas, hub_triplets):
    """% de la participación en hubs que corresponde a cada subred."""
    p = region_participation(atlas, hub_triplets)
    s = p.groupby("subred")["hubs"].sum().reindex(list(NETWORKS)).fillna(0)
    return (100 * s / max(s.sum(), 1)).round(1)
