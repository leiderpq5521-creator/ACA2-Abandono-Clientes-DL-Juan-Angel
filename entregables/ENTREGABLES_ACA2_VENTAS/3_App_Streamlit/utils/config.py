"""Configuración compartida entre el notebook y la app Streamlit (ACA N° 2 · Ventas)."""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DIR_DATA = BASE / "data"
DIR_MODELOS = BASE / "modelos"
DIR_FIGURAS = BASE / "figuras"
RUTA_DB = DIR_DATA / "ventas.db"
RUTA_RDA = DIR_DATA / "raw" / "onlineretail.rda"          # dataset crudo (no se sube a GitHub)
RUTA_ACA1 = DIR_DATA / "clientes_aca1.csv"                 # tabla de clientes exportada por el ACA 1 (verificación)

SEED = 42
FECHA_CORTE = pd.Timestamp("2011-09-01")
FECHA_FIN = pd.Timestamp("2011-12-09")
N_SEMANAS = 40           # semanas de historia antes del corte (1-dic-2010 → 31-ago-2011)

# Canales base (los que el usuario puede escribir en la app) y canales derivados
BASE_CANALES = ["gasto", "pedidos", "unidades", "productos", "devoluciones"]
DERIVADOS = ["dias_sin_comprar", "es_cliente"]
CANALES = BASE_CANALES + DERIVADOS
ESTATICAS = ["recencia", "frecuencia", "monetario"]       # R-F-M derivados de la propia secuencia
ETIQUETAS = {
    "gasto": "Gasto (£)", "pedidos": "Pedidos", "unidades": "Unidades", "productos": "Productos distintos",
    "devoluciones": "Devoluciones", "dias_sin_comprar": "Días sin comprar", "es_cliente": "Ya es cliente",
    "recencia": "Recencia (días)", "frecuencia": "Frecuencia (pedidos)", "monetario": "Gasto total (£)",
}

# Orden de las clases = salida de la red (logit de la clase 1)
CLASES = ["Recompra", "Abandono"]
COLORES = {"Recompra": "#1B7F79", "Abandono": "#E07A1F"}
COLOR_MODELO = {"Random Forest (ACA 1)": "#8AA0B8", "GRU": "#2E86AB", "LSTM": "#6A4C93", "Transformer": "#E07A1F"}


def inicio_semana(t: int) -> pd.Timestamp:
    """Fecha del primer día de la semana t (0 = la más antigua, N_SEMANAS-1 = la última antes del corte)."""
    return FECHA_CORTE - pd.Timedelta(days=7 * (N_SEMANAS - t))
