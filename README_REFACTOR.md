# Reorganización del código HOI

Paquete `hoi/` con lo que estaba duplicado o era lento en los notebooks. No cambia ningún resultado:
los tests comparan contra el código original.

| Módulo | Sustituye a | Nota |
|---|---|---|
| `hoi.hypergraph` | `create_hypergraph`, `HO_cent_df` (bloque 4 y `surrogate_analysis.py`) | adyacencia con producto de incidencias (~100x más rápida) |
| `hoi.nullmodel` | `parallel_process`, `generate_and_compute_centrality` | reproducible con `seed` |
| `hoi.information` | bucle de `soinfo_from_covmat` en `HOI_connectivity.py` | vectorizado (~100x), mismo resultado |
| `hoi.pipeline` | `run_all_HOI`, `run_High_Order`, `info_topo` (bloque 1) | mismas columnas y orden; O/S-info vectorizada, entropías con `infotopo` sin cambios |
| `scripts/run_block1.py` | bucle manual por sujetos | varios sujetos en paralelo y reanudable |
| `hoi.group` | notebook del bloque 3 (normalización, promedio, tripletes significativos) | un solo código para real y aleatorio; memoria constante; valida el orden de los tripletes |
| `scripts/run_block3.py` | celdas repetidas | CSV por sujeto -> promedios, con avisos |
| `hoi.report`, `plots.plot_adjacency`, `plots.animate_hubs_3d` | notebook del bloque 4 | tabla de hubs por región/subred, matriz de adyacencia (sin clustermap) y GIF giratorio |
| `hoi.pairwise` | notebook del bloque 5 (promedio de |corr|, umbral, matrices, centralidades, hubs) | verificado idéntico al original; `scripts/run_block5.py` y opción `--avg-corr` en el visor y las figuras |
| `hoi.surrogates` | `phase_randomization`, `phase_randomization_dataframe` (bloque 2) | **corrige** la orientación y conserva el espectro; incluye `mode='legacy'` para reproducir la función original |
| `scripts/run_block2.py` | bucle manual por sujetos | un surrogate por sujeto, reproducible (semilla) y reanudable |
| `scripts/run_null_model.py` | `surrogate_analysis.py` | misma salida, con argumentos de línea de comandos |

## Bloque 1: dónde está el tiempo
Con 92 regiones y 1200 instantes, por sujeto: O/S-info 14,2 s -> 0,15 s, pero las entropías de `infotopo`
(sin tocar) tardan ~190 s. La ganancia real viene de paralelizar entre sujetos (`--workers`).
Acelerar `infotopo` exigiría reimplementar su estimador discretizado y verificarlo contra él.
Cambios de comportamiento respecto al original: se eliminan las filas con NaN (antes: siempre la última fila),
y se comprueba que infotopo y O/S-info listan los tripletes en el mismo orden (antes se asumía).

## Bloque 1 con datos reales (serie de ejemplo del repo)
`compute_hoi` sobre `AAL_timeseries_example.txt` (1196 x 92) frente a `example_triplets.csv`: mismos 125.580 tripletes y
orden; Joint Ent, Mut Info y Total Corr idénticos (~1e-16). Oinfo/Sinfo difieren hasta 5e-6 / 2e-5 del CSV guardado,
pero el código ORIGINAL ejecutado hoy da esa misma diferencia contra el CSV y coincide con el nuevo hasta ~1e-15:
el CSV se generó con otra versión de las librerías (no se ha identificado cuál).

## Empates con el umbral
`hoi.exceeds` ignora los valores que solo superan el umbral por ruido de coma flotante (DC es discreto: 3/626 se puede
calcular de dos formas que difieren en el último bit). Tolerancia relativa 1e-9; no afecta a EC bruto (umbral ~1e-15).

## Tests
    HOI_DATA_DIR=<carpeta con los CSV> HOI_LEGACY_DIR=<carpeta con HOI_connectivity.py> python -m pytest tests

Comprueban: bloque 1 completo idéntico a `run_all_HOI` (datos sintéticos), adyacencia y centralidades idénticas a las originales (redundancia y sinergia), O/S-información
idéntica a `high_order`, recuentos de hubs (z-score, EC, p=0,05: 140 / 219 / 80 / 150), y conservación del
espectro en la aleatorización de fase.

## Cosas que NO se han decidido aquí (preguntar al supervisor)
1. **Dos CSV de tripletes distintos** (`average_triplets.csv` y `average_triplets_1.csv`): mismos tripletes,
   valores distintos; el 0,5 % superior solo coincide en 504/627 (redundancia) y 294/627 (sinergia).
2. **Aleatorización de fase del bloque 2** (comprobado con `example.txt` / `example_pr.csv`): la función original procesa cada FILA
   (un instante, 92 regiones), no cada serie temporal. Resultado en tu `example_pr.csv`: autocorrelación 0,47 -> 0,00; correlación
   media entre regiones 0,30 -> 0,97; TODOS los tripletes redundantes (O-info 0,65-0,97, por encima del máximo real 0,64).
   La versión nueva aleatoriza cada región en el tiempo conservando su espectro (error ~1e-14): `independent` da O-info ~0 (nulo sin
   dependencias) y `shared` reproduce la distribución real. **Si los resultados del artículo se hicieron con la función original,
   cambiarían; hay que preguntar cuál se usó.**
3. **Criterio de hub**: valor bruto (umbral nulo de EC ≈ 0, casi todo es "hub"), z-score o umbral fijo 0,12.
4. **Datos por sujeto en Git LFS**: sin `git lfs pull` los CSV son punteros y el bloque 3 no se puede reproducir.

## Bloque 3: qué sale de leer el notebook y los CSV
- **Los dos CSV de tripletes**: la celda 0 del notebook (carpeta `CodeBlock3` del repo) coincide exactamente con `average_triplets_1.csv`.
  El otro (`average_triplets.csv`) viene de otro sitio. El notebook carga por defecto `Results_RL`; el bloque 4 dice que su CSV es LR.
- **Verificado con datos reales**: `average_triplets_1.csv` (carpeta CodeBlock3) + `average_triplets_random.csv` reproducen EXACTAMENTE los
  recuentos guardados en el notebook (Mut Info, z-score, criterio 2,5/97,5): 68.458 de 125.580 (54,5 %) = 28.685 redundantes + 39.773
  sinérgicos. Con el otro promedio (`average_triplets.csv`) salen 79.365, así que NO es la pareja de ese aleatorio.
  Resumen por métrica (z-score): Joint Ent 49,4 %, Mut Info 54,5 %, Total Corr 50,3 %, Oinfo 56,6 %, Sinfo 39,3 %.
- **Dispersión**: el promedio aleatorio es mucho más estrecho que el real (std de Mut Info z-score 0,12 frente a 0,48), como cabe esperar si
  cada sujeto tiene un surrogate independiente que se promedia. Por eso el criterio marca tantos tripletes: mide consistencia entre
  sujetos por encima del azar, no una probabilidad por triplete (el propio notebook lo advierte como aproximación).
- **Este paso no alimenta el bloque 4** tal como lo he leído: el hipergrafo se construye con el 0,5 % superior por VALOR y el nulo de
  centralidad usa tripletes elegidos al azar, no los surrogate. Hay que confirmar si los significativos se usan en algún otro sitio.
- `Category` ('Synergistic'/'Redundant') solo tiene sentido para métricas con signo (Mut Info, Oinfo).
- El código original no comprobaba que real y surrogate tienen el mismo nº de sujetos, la misma adquisición ni el mismo orden de tripletes.

## Bloque 4: verificado contra las salidas guardadas del notebook original
El notebook original guarda (en sus salidas) los umbrales del modelo nulo, las primeras filas de las centralidades reales y el nº de hubs.
Con `hoi`: los 6 umbrales (EC, EC_z, BC, BC_z, DC, DC_z; p = 0,05) son IDÉNTICOS; las 6 centralidades de las 5 primeras filas coinciden
(redundancia con Mut Info_normalized y con Total Corr_zscore, error relativo < 6e-5, límite de las cifras impresas); y con el criterio bruto
`EC > EC_sig` salen 627 hubs, lo que imprime el notebook. Esto confirma además que el bloque 4 usa `average_triplets.csv` (no `_1`).
- **Tres criterios de hub** en el notebook original: z-score (`EC_z > umbral`), valor bruto (`EC > umbral`: umbral ~6,5e-15, TODOS los
  tripletes son hubs) y umbral fijo 0,12 (33 tripletes). Falta decidir cuál es el del artículo.
- **No cubierto**: gráficos de cuartetos (k = 4) y las animaciones/HTML de plotly (`plotho3dk_sub`, 2.600 líneas en `HOI_data_vis.py`);
  el visor HTML y el GIF las sustituyen en parte.

## Manuscrito v1 (bioRxiv 2023.02.10.528083v1): reproducido con `average_triplets.csv`
El manuscrito usa los **1000 tripletes más fuertes** (no el 0,5 %) y llama hubs a los de EC "lejos de cero". Con `n_triplets=1000`:
- **Información de interacción, redundancia**: EC > 0,12 -> **42 hubs (el artículo dice 42)**, los 42 con Precentral_L y Postcentral_L.
- **Correlación total, redundancia** (Total Corr_zscore): hubs con ambos cuneus (90 con EC > 0,05; el máximo de EC es 0,108, por lo que 0,12 no selecciona ninguno).
- **Sinergia** (los 1000 más negativos): hubs con el giro angular izquierdo (166 con EC > 0,05; accumbens derecho 57, izquierdo 55; con EC > 0,12 solo 2).
- Con `average_triplets_1.csv` salen 41 hubs y el accumbens izquierdo: **el conjunto del artículo (y del bloque 4) es `average_triplets.csv`**.
Esto valida de punta a punta (selección, adyacencia, EC) contra el texto del artículo. El umbral concreto de TC y sinergia no lo da el texto que he leído
(solo "una pequeña fracción con EC no nulo"), así que 0,05 es una elección mía. La versión actual del repositorio (0,5 % + modelo nulo) es posterior a v1.
Ojo: `aggregated_centralities1000copies0001fraction.csv` (nombre en el notebook) es casi seguro `...001fraction.csv` (1255 tripletes por copia = fracción 0,01);
el nulo depende del tamaño del hipergrafo (umbral de EC p=0,05: ~6e-15 con 627 tripletes y 0,0065 con 1255). El selector p del visor es el percentil del nulo de 627, NO esa variante.

## Bloque 5 (análisis por pares): verificado y dos cuestiones abiertas
Comprobado contra el código original del notebook con tramos reales de tu serie de ejemplo (3 "sujetos" = 3 tramos de UNA serie, no un grupo): matriz
promedio (2,7e-14), umbral, matrices binarizada y ponderada, y las 5 listas de hubs (EC, DC, BC binarizadas; EC, BC ponderadas) IDÉNTICAS en nodos y orden.
No se ha podido comprobar con el grupo real: hacen falta las series de los 50-100 sujetos (o la matriz promedio) que el repo no trae.
- **Diagonal en el percentil**: el original calcula el percentil 95 con la matriz aplanada, que incluye la diagonal (92 unos) y cada par dos veces, y deja los
  autoenlaces. Los autoenlaces por sí solos no cambian ninguna centralidad (~1e-16), pero el umbral sube (en la prueba: 0,754 frente a 0,736) y quedan 166 enlaces
  en vez de 209 (~20 % menos). En la prueba los conjuntos de hubs cambian en parte (p. ej. DC: 10 frente a 7, 6 comunes). `include_diagonal=False` lo corrige.
- **Betweenness ponderada**: networkx usa `weight` como DISTANCIA; el original le pasa la fuerza de la correlación (más correlación = más lejos). Se mantiene por
  fidelidad y está documentado; para una BC con sentido habría que usar distancias (p. ej. 1/peso).
- **Nombres**: el notebook lee `AAL92_region_names_full.txt` con tabuladores y trunca `Frontal_Inf_Tri_L` y `Precuneus_L` (salen sin la L). `hoi.atlas` los lee limpios.
- Los archivos `unrestricted_*.xlsx` y `BD.csv` (datos conductuales/anatómicos de HCP) no son del bloque 5 sino del 6 y NO van en el zip.

## Corrección de un error propio (versión anterior del zip)
El modo `independent=False` de `hoi.surrogates` sustituía las fases por fases comunes: no conservaba la correlación entre regiones
(0,80 en vez de 0,30) aunque su descripción decía que sí, y su test no lo detectó porque usaba datos casi perfectamente
correlacionados. Ahora SUMA un desfase común a las fases originales y el test exige igualdad de la matriz de correlación (1e-9).

## Compatibilidad
`networkx.eigenvector_centrality_numpy` lanza `AmbiguousSolution` en grafos desconectados (networkx >= 3), y los
hipergrafos aleatorios del modelo nulo lo son; `surrogate_analysis.py` se rompería con un entorno moderno.
`hoi.hypergraph.eigenvector_centrality` calcula el mismo vector sin ese error.
