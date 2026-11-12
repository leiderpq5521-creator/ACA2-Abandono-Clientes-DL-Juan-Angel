"""Inferencia y explicación individual: el MISMO preprocesamiento que usa el notebook (Escalador.transformar)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .config import BASE_CANALES, N_SEMANAS
from .secuencias import Escalador


def preparar(base: np.ndarray, esc: Escalador) -> tuple[torch.Tensor, torch.Tensor]:
    """(T, 5) o (n, T, 5) canales base en unidades originales → tensores escalados (n, T, 7) y (n, 3)."""
    seq, est = esc.transformar(np.asarray(base, dtype=np.float32))
    return torch.from_numpy(seq), torch.from_numpy(est)


@torch.inference_mode()
def probabilidad(modelo, esc: Escalador, base: np.ndarray, lote: int = 512) -> np.ndarray:
    base = np.asarray(base, dtype=np.float32)
    if base.ndim == 2:
        base = base[None]
    modelo.eval()
    out = [torch.sigmoid(modelo(*preparar(base[i:i + lote], esc))).numpy() for i in range(0, len(base), lote)]
    return np.concatenate(out)


@torch.inference_mode()
def oclusion_semanas(modelo, esc: Escalador, base: np.ndarray) -> tuple[pd.DataFrame, float]:
    """Para cada semana con actividad: cambio en P(abandono) si esa semana se borra (oclusión).
    Positivo = esa semana REDUCÍA el riesgo (borrarla lo sube)."""
    base = np.asarray(base, dtype=np.float32)
    activas = np.flatnonzero((base[:, 1] > 0) | (base[:, 4] > 0))
    p0 = float(probabilidad(modelo, esc, base)[0])
    if len(activas) == 0:
        return pd.DataFrame(columns=["semana", "efecto"]), p0
    variantes = np.repeat(base[None], len(activas), axis=0)
    for k, t in enumerate(activas):
        variantes[k, t] = 0
    validas = (variantes[:, :, 1] > 0).any(1)            # sin ningún pedido la secuencia no tiene sentido
    p = np.full(len(activas), np.nan)
    if validas.any():
        p[validas] = probabilidad(modelo, esc, variantes[validas])
    return pd.DataFrame({"semana": activas, "efecto": p - p0}), p0


def secuencia_desde_tabla(df: pd.DataFrame) -> np.ndarray:
    """Tabla con columnas BASE_CANALES (una fila por semana, la última = semana previa al corte) → (T, 5).
    Si tiene menos de N_SEMANAS filas se rellena con semanas vacías al inicio; si tiene más, se usan las últimas."""
    faltan = [c for c in BASE_CANALES if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas: {', '.join(faltan)}")
    # to_numeric con errors="coerce": un texto o celda vacía en la tabla editable cuenta como 0 en vez de fallar
    x = df[BASE_CANALES].apply(pd.to_numeric, errors="coerce").fillna(0).clip(lower=0).to_numpy(np.float32)[-N_SEMANAS:]
    base = np.zeros((N_SEMANAS, len(BASE_CANALES)), dtype=np.float32)
    base[N_SEMANAS - len(x):] = x
    return base

