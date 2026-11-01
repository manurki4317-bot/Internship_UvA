# Reorganización del repositorio HOI

Qué hay en esta carpeta y cómo usarlo. Las decisiones pendientes están en `README_REFACTOR.md`.

```
hoi/                  paquete con la lógica (hipergrafo, centralidades, modelo nulo, O/S-info, bloque 1, gráficos)
HOI_connectivity.py   bloque 1 reorganizado: mismos nombres públicos que el original, con cada cambio comentado ("CAMBIO:")
scripts/              run_block1.py, run_block2.py, run_block3.py y run_block5.py (varios sujetos en paralelo), run_null_model.py (sustituye a surrogate_analysis.py)
viz/                  visor interactivo: hoi_hub_viewer.html (ya construido), build_viewer.py, make_figures.py
figures/              4 figuras estáticas de ejemplo (PNG), generadas con tus datos
atlas/                AAL92 (nombres, posiciones, subredes, colores) y malla simplificada del cerebro
notebooks/            Block1_Computing_HOI.ipynb, Block2_Phase_Randomization.ipynb, Block3_Hyperedge_selection.ipynb, Block4_Hubs.ipynb, Block5_Pairwise.ipynb y Visualization.ipynb (cortos: llaman al paquete)
tests/                comparan el paquete con el código original
```

## Gráficos
- **Interactivo (3D)**: abre `viz/hoi_hub_viewer.html` en el navegador (necesita internet: carga Plotly de un CDN).
  Selector de métrica, redundancia/sinergia, tamaño del hipergrafo (627 con nulo o 1000 como el artículo v1), centralidad, criterio (z-score/bruto/EC manual), p, vista de cerebro o de pares de
  regiones, y hubs por pares marcados con un aro ámbar. Para regenerarlo con otros CSV:
  `python viz/build_viewer.py --triplets <average_triplets.csv> --null <aggregated_centralities...csv>`
- **Estático (PNG)**: `python viz/make_figures.py --triplets ... --null ... --out figures` genera proyecciones
  axial/coronal/sagital de los hubs, matriz de pares de regiones, nulo frente a real y por pares frente a alto orden.
  En notebook: `notebooks/Visualization.ipynb` (usa `hoi.plots`).

## Bloque 5
`notebooks/Block5_Pairwise.ipynb` y `scripts/run_block5.py`: matriz promedio de |correlación|, hubs por pares y comparación con alto orden. Idéntico al original en las pruebas (ver `README_REFACTOR.md`, con dos cuestiones abiertas).

## Bloque 4
`notebooks/Block4_Hubs.ipynb`: hipergrafo, centralidades, hubs, tablas, figuras y GIF con `hoi`. Verificado contra las cifras guardadas en el notebook original y contra el manuscrito v1 (42 hubs con `MODE='paper_v1'`; ver `README_REFACTOR.md`). Los 3D interactivos están en `viz/`.

## Bloque 3
`hoi.average_subjects(files)` y `hoi.find_significant_triplets(real, random, 'Mut Info')`. Mismos resultados que el notebook; ahora comprueba orden de tripletes y avisa si los CSV son punteros de Git LFS.

## Bloque 2
`hoi.make_surrogate(df, mode='independent')`. OJO: cambia el surrogate respecto al código original (ver `README_REFACTOR.md`, punto 2). `mode='legacy'` reproduce el original.

## Bloque 1
`from HOI_connectivity import run_all_HOI` funciona igual que antes (o `hoi.compute_hoi(df)`). Necesita
`infotopo_server.py` en la misma carpeta o en el `PYTHONPATH` (no se incluye: es la librería de Baudot).

## Tests
`HOI_DATA_DIR=<csv> HOI_LEGACY_DIR=<HOI_connectivity.py original + infotopo_server.py> HOI_EXAMPLE_DIR=<serie y CSV de ejemplo> python -m pytest tests`
(sin las variables, los tests que las necesitan se omiten).
