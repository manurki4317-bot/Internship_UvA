"""Series temporales surrogate por aleatorización de fase (bloque 2).

Método: X'(f) = X(f) * exp(i*theta(f)), con theta uniforme en [0, 2pi), theta=0 en la componente continua y en la
frecuencia de Nyquist, y simetría conjugada (se usa rfft/irfft, así la serie resultante es real sin descartar nada).
  - El espectro de amplitud de cada serie se conserva EXACTAMENTE (error ~1e-14), por tanto también su
    autocorrelación y su media.
  - modo 'independent'  : theta distinta en cada región -> se destruye la dependencia entre regiones.
  - modo 'shared'       : la MISMA theta en todas las regiones -> se conserva el espectro cruzado y, por tanto,
                          la matriz de correlación entre regiones (surrogate "multivariante" clásico).
  - modo 'legacy'       : reproduce la función original del bloque 2 (ver más abajo). SOLO para comparar.

CORRECCIÓN respecto a la versión anterior de este archivo: 'shared' sustituía las fases por fases comunes (todas
las regiones quedaban en fase y la correlación se disparaba); lo correcto es SUMAR un desfase común a las fases originales.
"""
import numpy as np
import pandas as pd

MODES = ("independent", "shared", "legacy")


def phase_randomize(data, axis=0, independent=True, rng=None):
    """Aleatoriza la fase de series a lo largo de `axis` (el eje del tiempo). Ver el modo en la cabecera del módulo."""
    rng = np.random.default_rng(rng)
    x = np.moveaxis(np.asarray(data, dtype=float), axis, 0)
    T = x.shape[0]
    F = np.fft.rfft(x, axis=0)
    shape = F.shape if independent else (F.shape[0],) + (1,) * (F.ndim - 1)
    theta = rng.uniform(0, 2 * np.pi, size=shape)
    theta[0] = 0                       # componente continua: se conserva la media
    if T % 2 == 0:
        theta[-1] = 0                  # Nyquist: debe ser real
    y = np.fft.irfft(F * np.exp(1j * theta), n=T, axis=0)
    return np.moveaxis(y, 0, axis)


def phase_randomize_legacy(df, rng=None):
    """Función ORIGINAL del bloque 2, tal cual: procesa cada FILA de `df` (en tiempo x regiones, un instante con todas las
    regiones), baraja las fases de esa fila y descarta la parte imaginaria. No conserva el espectro, destruye la
    autocorrelación temporal y deja a las regiones casi perfectamente correlacionadas. Solo para reproducir/comparar."""
    rng = np.random.default_rng(rng)
    out = []
    for _, row in df.iterrows():
        f = np.fft.fft(row.values)
        ph = np.angle(f)
        rng.shuffle(ph)
        out.append(pd.Series(np.fft.ifft(np.abs(f) * np.exp(1j * ph)).real))
    res = pd.concat(out, axis=1).transpose()
    res.columns = df.columns
    return res


def make_surrogate(df, mode="independent", seed=None):
    """Surrogate de un sujeto. df: tiempo x regiones (se descartan las filas con NaN, como en el bloque 1).
    Devuelve un DataFrame con la misma forma y columnas."""
    if mode not in MODES:
        raise ValueError(f"mode debe ser uno de {MODES}")
    clean = df.dropna()
    if mode == "legacy":
        out = phase_randomize_legacy(clean, rng=seed)
        out.index = clean.index
        return out
    y = phase_randomize(clean.to_numpy(float), axis=0, independent=(mode == "independent"), rng=seed)
    return pd.DataFrame(y, index=clean.index, columns=clean.columns)
