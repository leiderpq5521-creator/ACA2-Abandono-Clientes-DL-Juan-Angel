"""Utilidades compartidas entre el notebook y la app Streamlit (ACA N° 1 – Ventas)."""

import numpy as np
import pandas as pd

# Variables del modelo (nivel cliente) con etiqueta y descripción en español
VARIABLES = {
    "recencia_dias": ("Recencia (días)", "Días transcurridos desde la última compra hasta la fecha de corte"),
    "frecuencia": ("Frecuencia (pedidos)", "Número de pedidos (facturas) distintos realizados"),
    "monetario": ("Gasto total (£)", "Suma del importe de todas las compras"),
    "ticket_promedio": ("Ticket promedio (£)", "Gasto total dividido por el número de pedidos"),
    "n_productos": ("Productos distintos", "Cantidad de referencias (StockCode) distintas compradas"),
    "unidades_por_pedido": ("Unidades por pedido", "Promedio de unidades por pedido"),
    "antiguedad_dias": ("Antigüedad (días)", "Días desde la primera compra hasta la fecha de corte"),
    "dias_entre_compras": ("Días entre compras", "Promedio de días entre pedidos consecutivos (vacío si solo hay uno)"),
    "compras_90d": ("Pedidos últimos 90 días", "Pedidos en los 90 días previos al corte"),
    "gasto_90d": ("Gasto últimos 90 días (£)", "Gasto en los 90 días previos al corte"),
    "n_devoluciones": ("Devoluciones", "Número de facturas de cancelación/devolución del cliente"),
    "es_uk": ("Cliente del Reino Unido", "1 si el país del cliente es Reino Unido, 0 en otro caso"),
}
FEATURES = list(VARIABLES)
# Variables muy asimétricas (cola larga a la derecha): se transforman con log(1+x) dentro del pipeline
LOG_VARS = ["frecuencia", "monetario", "ticket_promedio", "n_productos", "unidades_por_pedido",
            "dias_entre_compras", "compras_90d", "gasto_90d", "n_devoluciones"]
RFM = ["recencia_dias", "frecuencia", "monetario"]


def etiqueta(col: str) -> str:
    return VARIABLES[col][0] if col in VARIABLES else col


def descripcion(col: str) -> str:
    return VARIABLES[col][1] if col in VARIABLES else ""


# Nombres de segmentos, ordenados de menor a mayor valor del cliente (para mostrar de mayor a menor se invierte)
NOMBRES_SEGMENTOS = {
    3: ["Dormidos", "Recientes", "Campeones"],
    4: ["Dormidos", "En riesgo", "Recientes", "Campeones"],
    5: ["Dormidos", "En riesgo", "Ocasionales", "Recientes", "Campeones"],
}
_INTERMEDIOS = {1: ["Recientes"], 2: ["Recientes", "En riesgo"], 3: ["Recientes", "Ocasionales", "En riesgo"]}


def nombrar_segmentos(centros) -> dict:
    """Asigna nombre a cada cluster a partir de sus centroides estandarizados (columnas R, F, M).
    Campeones = mayor frecuencia+gasto con baja recencia; Dormidos = mayor recencia entre el resto;
    los intermedios se ordenan por recencia (menor recencia = 'Recientes')."""
    c = pd.DataFrame(centros, columns=["R", "F", "M"])
    campeon = (c.F + c.M - c.R).idxmax()
    resto = [i for i in c.index if i != campeon]
    dormido = c.loc[resto, "R"].idxmax()
    medios = c.loc[[i for i in resto if i != dormido]].sort_values("R").index.tolist()
    nombres = {campeon: "Campeones", dormido: "Dormidos"}
    intermedios = _INTERMEDIOS.get(len(medios), [f"Intermedio {j + 1}" for j in range(len(medios))])
    for i, n in zip(medios, intermedios):
        nombres[i] = n
    return nombres


def log_rfm(df: pd.DataFrame) -> np.ndarray:
    """Espacio de segmentación: log(recencia), log(frecuencia), log(gasto)."""
    return np.column_stack([np.log1p(df["recencia_dias"]), np.log1p(df["frecuencia"]), np.log1p(df["monetario"])])
