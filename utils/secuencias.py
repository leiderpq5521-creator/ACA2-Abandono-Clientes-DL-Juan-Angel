"""Ingeniería de secuencias semanales por cliente, escalado y aumento de datos (PASO 2 de la ruta).

Una sola función (`derivar`) convierte los 5 canales base de cualquier secuencia —venga de la base de datos,
del editor de la app o de un CSV— en los 7 canales de entrada y las 3 variables R-F-M. Así el notebook y la app
preprocesan exactamente igual.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import BASE_CANALES, CANALES, ESTATICAS, FECHA_CORTE, N_SEMANAS

LOG_CANALES = [c for c in CANALES if c != "es_cliente"]      # canales con cola larga → log(1+x)


# ───────────────────────── Construcción desde las transacciones ─────────────────────────
def semana_de(fechas: pd.Series) -> np.ndarray:
    """Índice de semana 0…N_SEMANAS-1 (N_SEMANAS-1 = los 7 días previos al corte). Fechas ≥ corte → error."""
    dias = (FECHA_CORTE - fechas.dt.normalize()).dt.days.to_numpy()
    if (dias <= 0).any():
        raise ValueError("Hay fechas en o después del corte: fuga de información futura")
    return N_SEMANAS - 1 - (dias - 1) // 7


def construir_series(lineas: pd.DataFrame, cancelaciones: pd.DataFrame) -> pd.DataFrame:
    """Serie semanal en formato largo (CustomerID, semana, 5 canales base) a partir de líneas de factura limpias
    (sólo ventas con fecha < corte) y de las facturas de cancelación del mismo periodo."""
    lineas = lineas.assign(semana=semana_de(lineas.InvoiceDate), importe=lineas.Quantity * lineas.UnitPrice)
    ventas = lineas.groupby(["CustomerID", "semana"]).agg(
        gasto=("importe", "sum"), pedidos=("InvoiceNo", "nunique"), unidades=("Quantity", "sum"),
        productos=("StockCode", "nunique"))
    canc = cancelaciones.assign(semana=semana_de(cancelaciones.InvoiceDate))
    dev = canc.groupby(["CustomerID", "semana"]).InvoiceNo.nunique().rename("devoluciones")
    serie = ventas.join(dev, how="outer").fillna(0).reset_index()
    serie["CustomerID"] = serie.CustomerID.astype(int)
    serie["semana"] = serie.semana.astype(int)
    return serie[["CustomerID", "semana"] + BASE_CANALES]


def a_tensor_base(series: pd.DataFrame, ids) -> np.ndarray:
    """Formato largo → arreglo denso (n_clientes, N_SEMANAS, 5) en el orden de `ids` (semanas sin actividad = 0)."""
    ids = np.asarray(ids)
    pos = pd.Series(np.arange(len(ids)), index=ids)
    s = series[series.CustomerID.isin(ids)]
    base = np.zeros((len(ids), N_SEMANAS, len(BASE_CANALES)), dtype=np.float32)
    base[pos[s.CustomerID].to_numpy(), s.semana.to_numpy()] = s[BASE_CANALES].to_numpy(dtype=np.float32)
    return base


# ───────────────────────── Canales derivados (mismo código en notebook y app) ────────────
def derivar(base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n, T, 5) canales base → (n, T, 7) canales completos y (n, 3) estáticas R-F-M, en unidades originales.
    días_sin_comprar(t) = 7 × semanas desde la última semana con pedido (0 antes de la primera compra)."""
    base = np.asarray(base, dtype=np.float32)
    if base.ndim == 2:
        base = base[None]
    n, T, _ = base.shape
    activo = base[..., BASE_CANALES.index("pedidos")] > 0
    t = np.broadcast_to(np.arange(T), (n, T))
    ultima = np.maximum.accumulate(np.where(activo, t, -1), axis=1)
    es_cliente = (ultima >= 0).astype(np.float32)
    dias = np.where(ultima >= 0, 7.0 * (t - ultima), 0.0).astype(np.float32)
    completo = np.concatenate([base, dias[..., None], es_cliente[..., None]], axis=-1)
    estaticas = np.stack([dias[:, -1], base[..., 1].sum(1), base[..., 0].sum(1)], axis=1).astype(np.float32)
    return completo, estaticas


class Escalador:
    """log(1+x) en los canales asimétricos + estandarización con media/desv. calculadas SÓLO en entrenamiento."""

    def __init__(self, media_seq=None, desv_seq=None, media_est=None, desv_est=None):
        self.media_seq, self.desv_seq = media_seq, desv_seq
        self.media_est, self.desv_est = media_est, desv_est
        self.idx_log = [CANALES.index(c) for c in LOG_CANALES]

    def _log(self, seq, est):
        seq = seq.copy(); seq[..., self.idx_log] = np.log1p(np.maximum(seq[..., self.idx_log], 0))
        return seq, np.log1p(np.maximum(est, 0))

    def ajustar(self, base_train: np.ndarray) -> "Escalador":
        seq, est = self._log(*derivar(base_train))
        self.media_seq = seq.reshape(-1, seq.shape[-1]).mean(0)
        self.desv_seq = seq.reshape(-1, seq.shape[-1]).std(0) + 1e-6
        i = CANALES.index("es_cliente")                  # el indicador 0/1 se deja tal cual
        self.media_seq[i], self.desv_seq[i] = 0.0, 1.0
        self.media_est, self.desv_est = est.mean(0), est.std(0) + 1e-6
        return self

    def transformar(self, base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        seq, est = self._log(*derivar(base))
        return (((seq - self.media_seq) / self.desv_seq).astype(np.float32),
                ((est - self.media_est) / self.desv_est).astype(np.float32))

    def a_dict(self) -> dict:
        return {"canales": CANALES, "estaticas": ESTATICAS, "log_canales": LOG_CANALES, "n_semanas": N_SEMANAS,
                **{k: np.asarray(getattr(self, k)).round(8).tolist() for k in ("media_seq", "desv_seq", "media_est", "desv_est")}}

    def guardar(self, ruta) -> None:
        ruta.write_text(json.dumps(self.a_dict(), indent=1), encoding="utf-8")

    @classmethod
    def cargar(cls, ruta) -> "Escalador":
        d = json.loads(ruta.read_text(encoding="utf-8"))
        assert d["canales"] == CANALES and d["n_semanas"] == N_SEMANAS, "Escalador incompatible con la configuración"
        return cls(*(np.array(d[k], dtype=np.float32) for k in ("media_seq", "desv_seq", "media_est", "desv_est")))


# ───────────────────────── Aumento de datos para series (sólo entrenamiento) ─────────────
def aumentar(base: np.ndarray, rng: np.random.Generator, p: float = 0.5) -> np.ndarray:
    """Aumento de una secuencia base (T, 5) en unidades originales. Se aplica antes de derivar, de modo que
    días_sin_comprar y R-F-M se recalculan de forma coherente.
    - Escalado aleatorio: gasto y unidades × factor log-normal (σ = 0,15) común a toda la secuencia.
    - Jitter: ruido multiplicativo (σ = 0,10) en cada semana activa (nunca crea actividad donde no la hay).
    - Recorte de ventana: se borran las k semanas más antiguas (simula un cliente más nuevo) sin vaciar la serie."""
    x = base.copy()
    gi, ui = BASE_CANALES.index("gasto"), BASE_CANALES.index("unidades")
    if rng.random() < p:
        x[:, [gi, ui]] *= np.exp(rng.normal(0, 0.15, size=2)).astype(np.float32)
    if rng.random() < p:
        activo = x[:, 1] > 0
        ruido = np.exp(rng.normal(0, 0.10, size=(len(x), 2))).astype(np.float32)
        x[np.ix_(activo, [gi, ui])] *= ruido[activo]
    if rng.random() < p * 0.6:
        semanas_activas = np.flatnonzero(x[:, 1] > 0)
        if len(semanas_activas) > 1:                      # deja al menos la última semana con compra
            k = int(rng.integers(1, semanas_activas[-1] + 1))
            x[:k] = 0
    return x


class DatasetSecuencias(Dataset):
    """Devuelve (secuencia (T, 7), estáticas (3), etiqueta) ya escaladas; aplica aumento si `aumentar_datos`."""

    def __init__(self, base: np.ndarray, y: np.ndarray, escalador: Escalador, aumentar_datos: bool = False,
                 seed: int = 0):
        self.base, self.y, self.esc, self.aug = base, np.asarray(y, dtype=np.float32), escalador, aumentar_datos
        self.rng = np.random.default_rng(seed)
        if not aumentar_datos:                           # sin aumento se precalcula todo una sola vez
            self.seq, self.est = escalador.transformar(base)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        if self.aug:
            seq, est = self.esc.transformar(aumentar(self.base[i], self.rng))
            seq, est = seq[0], est[0]
        else:
            seq, est = self.seq[i], self.est[i]
        return torch.from_numpy(seq), torch.from_numpy(est), torch.tensor(self.y[i])
