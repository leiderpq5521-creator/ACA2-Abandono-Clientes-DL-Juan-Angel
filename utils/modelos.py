"""Arquitecturas neuronales para clasificar secuencias semanales de compra (PASO 3 de la ruta).

Las tres redes comparten la misma interfaz: forward(seq (B,T,C), est (B,3)) → logit (B,) de la clase "Abandono".
La parte secuencial lee la «película» del cliente; la cabeza concatena su resumen con R-F-M (modelo híbrido).
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .config import CANALES, ESTATICAS, N_SEMANAS


class Cabeza(nn.Module):
    def __init__(self, d_in: int, oculto: int = 32, dropout: float = 0.2, sesgo_inicial: float = 0.0):
        super().__init__()
        self.red = nn.Sequential(nn.Linear(d_in + len(ESTATICAS), oculto), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(oculto, 1))
        nn.init.normal_(self.red[-1].weight, 0, 0.01)       # salida inicial ≈ sesgo → pérdida inicial ≈ ln 2
        nn.init.constant_(self.red[-1].bias, sesgo_inicial)

    def forward(self, h, est):
        return self.red(torch.cat([h, est], dim=1)).squeeze(1)


class RedRecurrente(nn.Module):
    """GRU o LSTM de 2 capas; el último estado oculto resume la secuencia completa."""

    def __init__(self, tipo: str = "gru", oculto: int = 32, capas: int = 2, dropout: float = 0.2,
                 sesgo_inicial: float = 0.0):
        super().__init__()
        rnn = nn.GRU if tipo == "gru" else nn.LSTM
        self.tipo = tipo
        self.rnn = rnn(len(CANALES), oculto, num_layers=capas, batch_first=True, dropout=dropout)
        self.cabeza = Cabeza(oculto, dropout=dropout, sesgo_inicial=sesgo_inicial)

    def forward(self, seq, est):
        salida, _ = self.rnn(seq)
        return self.cabeza(salida[:, -1], est)


class BloqueEncoder(nn.Module):
    """Bloque Transformer pre-norm (atención multi-cabeza + MLP) que guarda los pesos de atención."""

    def __init__(self, d: int, cabezas: int, ff: int, dropout: float):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, cabezas, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)
        self.atencion = None

    def forward(self, x):
        h = self.n1(x)
        a, w = self.att(h, h, h, need_weights=True, average_attn_weights=True)
        self.atencion = w.detach()                       # (B, T+1, T+1), promedio de cabezas
        x = x + self.drop(a)
        return x + self.drop(self.mlp(self.n2(x)))


class TransformerTemporal(nn.Module):
    """Encoder Transformer temporal: proyección lineal de cada semana + posición aprendida + token [CLS]."""

    def __init__(self, d: int = 32, cabezas: int = 4, capas: int = 2, ff: int = 64, dropout: float = 0.2,
                 sesgo_inicial: float = 0.0):
        super().__init__()
        self.proy = nn.Linear(len(CANALES), d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, N_SEMANAS + 1, d) * 0.02)
        self.bloques = nn.ModuleList([BloqueEncoder(d, cabezas, ff, dropout) for _ in range(capas)])
        self.norm = nn.LayerNorm(d)
        self.cabeza = Cabeza(d, dropout=dropout, sesgo_inicial=sesgo_inicial)

    def forward(self, seq, est):
        x = self.proy(seq)
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], dim=1) + self.pos
        for b in self.bloques:
            x = b(x)
        return self.cabeza(self.norm(x[:, 0]), est)

    def atencion_cls(self) -> torch.Tensor:
        """Atención del token [CLS] sobre las semanas en la última capa (B, T)."""
        return self.bloques[-1].atencion[:, 0, 1:]


ARQUITECTURAS = {
    "GRU": lambda **kw: RedRecurrente("gru", **kw),
    "LSTM": lambda **kw: RedRecurrente("lstm", **kw),
    "Transformer": lambda **kw: TransformerTemporal(**kw),
}
DESCRIPCION = {
    "GRU": "Red recurrente GRU de 2 capas (32 unidades) + R-F-M",
    "LSTM": "Red recurrente LSTM de 2 capas (32 unidades) + R-F-M",
    "Transformer": "Encoder Transformer temporal (2 bloques, 4 cabezas, d = 32, token [CLS]) + R-F-M",
}


def crear(nombre: str, sesgo_inicial: float = 0.0, **kw) -> nn.Module:
    return ARQUITECTURAS[nombre](sesgo_inicial=sesgo_inicial, **kw)


class Ensamble(nn.Module):
    """Promedio de las probabilidades de varias redes de la misma arquitectura (una por semilla).
    Devuelve el logit de la probabilidad media para mantener la misma interfaz que una red individual."""

    def __init__(self, miembros: list[nn.Module]):
        super().__init__()
        self.miembros = nn.ModuleList(miembros)

    def forward(self, seq, est):
        p = torch.stack([torch.sigmoid(m(seq, est)) for m in self.miembros]).mean(0)
        return torch.logit(p.clamp(1e-6, 1 - 1e-6))

    def atencion_cls(self) -> torch.Tensor:
        return torch.stack([m.atencion_cls() for m in self.miembros]).mean(0)


def n_parametros(modelo: nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters())


def guardar(miembros: list[nn.Module], nombre: str, ruta, **extra) -> None:
    """Guarda los pesos de TODAS las semillas de una arquitectura en un solo archivo."""
    torch.save({"arquitectura": nombre, "state_dicts": [m.state_dict() for m in miembros], **extra}, ruta)


def cargar(ruta) -> tuple[Ensamble, dict]:
    ck = torch.load(ruta, map_location="cpu", weights_only=True)   # sólo tensores y tipos básicos
    miembros = []
    for sd in ck["state_dicts"]:
        m = crear(ck["arquitectura"])
        m.load_state_dict(sd)
        miembros.append(m)
    ens = Ensamble(miembros)
    ens.eval()
    return ens, ck


def logit(p: float) -> float:
    return math.log(p / (1 - p))
