"""Gestión del dataset y del historial de predicciones en SQLite (PASO 1 de la ruta)."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BASE_CANALES, RUTA_DB

ESQUEMA = """
CREATE TABLE IF NOT EXISTS clientes (
    CustomerID    INTEGER PRIMARY KEY,
    pais          TEXT,
    recencia_dias INTEGER,                     -- días exactos desde la última compra hasta el corte (ACA 1)
    frecuencia    INTEGER,                     -- pedidos antes del corte
    monetario     REAL,                        -- gasto total antes del corte (£)
    segmento      TEXT,                        -- segmento K-Means del ACA 1
    abandono      INTEGER NOT NULL CHECK (abandono IN (0,1)),   -- 1 = no volvió a comprar tras el corte
    particion     TEXT CHECK (particion IN ('train','val','test'))
);
CREATE TABLE IF NOT EXISTS series (
    CustomerID    INTEGER NOT NULL REFERENCES clientes(CustomerID),
    semana        INTEGER NOT NULL,            -- 0 = la más antigua … 39 = la última antes del corte
    gasto REAL, pedidos INTEGER, unidades REAL, productos INTEGER, devoluciones INTEGER,
    PRIMARY KEY (CustomerID, semana)
);
CREATE TABLE IF NOT EXISTS predicciones (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha         TEXT NOT NULL,
    origen        TEXT,                        -- cliente:<id>, manual:<plantilla> o lote:<archivo>
    CustomerID    INTEGER,
    modelo        TEXT,
    prob_abandono REAL,
    umbral        REAL,
    decision      TEXT
);
CREATE INDEX IF NOT EXISTS ix_cli_part ON clientes(particion);
CREATE INDEX IF NOT EXISTS ix_series_cli ON series(CustomerID);
CREATE INDEX IF NOT EXISTS ix_pred_cli ON predicciones(CustomerID);
CREATE INDEX IF NOT EXISTS ix_pred_fecha ON predicciones(fecha DESC);

"""


def conectar(ruta: Path | str = RUTA_DB) -> sqlite3.Connection:
    con = sqlite3.connect(ruta, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.executescript(ESQUEMA)
    return con


def consultar(con: sqlite3.Connection, sql: str, params: tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, con, params=params)


def guardar_tabla(con: sqlite3.Connection, nombre: str, df: pd.DataFrame) -> int:
    """Reemplaza el contenido de una tabla del esquema (conserva tipos, claves e índices)."""
    con.execute(f"DELETE FROM {nombre}")
    df.to_sql(nombre, con, if_exists="append", index=False)
    con.commit()
    return con.execute(f"SELECT COUNT(*) FROM {nombre}").fetchone()[0]


def asignar_particion(con: sqlite3.Connection, particion: dict[int, str]) -> None:
    con.executemany("UPDATE clientes SET particion = ? WHERE CustomerID = ?",
                    [(p, int(c)) for c, p in particion.items()])
    con.commit()


def serie_cliente(con: sqlite3.Connection, customer_id: int, n_semanas: int) -> np.ndarray:
    """Secuencia base (T, 5) de un cliente leída de la tabla `series` (semanas sin fila = 0)."""
    s = consultar(con, f"SELECT semana, {', '.join(BASE_CANALES)} FROM series WHERE CustomerID = ?", (int(customer_id),))
    base = np.zeros((n_semanas, len(BASE_CANALES)), dtype=np.float32)
    base[s.semana.to_numpy()] = s[BASE_CANALES].to_numpy(dtype=np.float32)
    return base


def registrar_prediccion(con: sqlite3.Connection, origen: str, customer_id, modelo: str, prob: float,
                         umbral: float, decision: str) -> None:
    con.execute("INSERT INTO predicciones (fecha, origen, CustomerID, modelo, prob_abandono, umbral, decision) "
                "VALUES (?,?,?,?,?,?,?)",
                (datetime.now().isoformat(timespec="seconds"), origen,
                 None if customer_id is None else int(customer_id), modelo, float(prob), float(umbral), decision))
    con.commit()


def registrar_predicciones_lote(con: sqlite3.Connection, filas: list[tuple]) -> None:
    """Inserta múltiples predicciones en una sola transacción."""
    if not filas:
        return
    con.executemany("INSERT INTO predicciones (fecha, origen, CustomerID, modelo, prob_abandono, umbral, decision) "
                    "VALUES (?,?,?,?,?,?,?)", filas)
    con.commit()


