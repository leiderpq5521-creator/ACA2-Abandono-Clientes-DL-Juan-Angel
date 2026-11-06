"""
App Streamlit – ACA N° 2 Aprendizaje Automático (CUN) · Pilar Ventas
Predicción del abandono de clientes a partir de su secuencia semanal de compras con redes neuronales
(GRU, LSTM y Transformer temporal) comparadas con el Random Forest del ACA 1. Historial en SQLite.

Ejecutar:  streamlit run app.py
"""
import hashlib
import json
import sqlite3

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

from utils import db
from utils.config import (BASE_CANALES, CLASES, COLOR_MODELO, COLORES, DIR_DATA, DIR_FIGURAS, DIR_MODELOS, ETIQUETAS,
                          N_SEMANAS, RUTA_DB, inicio_semana)
from utils.inferencia import oclusion_semanas, probabilidad, secuencia_desde_tabla
from utils.modelos import DESCRIPCION
from utils.modelos import cargar as cargar_red
from utils.secuencias import Escalador, derivar

st.set_page_config(page_title="Abandono de clientes con Deep Learning", page_icon=":material/shopping_cart:", layout="wide")
torch.set_num_threads(1)
RF = "Random Forest (ACA 1)"
ARCHIVO_RED = {"GRU": "gru_abandono.pt", "LSTM": "lstm_abandono.pt", "Transformer": "transformer_abandono.pt"}


# ───────────────────────────── Carga de artefactos ──────────────────────────────
def _version(ruta) -> int:
    """Huella del archivo: si se actualiza (p. ej. un nuevo push a Streamlit Cloud), la caché se invalida."""
    return ruta.stat().st_mtime_ns ^ ruta.stat().st_size


@st.cache_data
def cargar_metricas(_version_archivo: int):
    return json.loads((DIR_MODELOS / "metricas.json").read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="Cargando los modelos…")
def cargar_modelos(_version_archivos: int):
    esc = Escalador.cargar(DIR_MODELOS / "escalador.json")
    redes = {n: cargar_red(DIR_MODELOS / f)[0] for n, f in ARCHIVO_RED.items() if (DIR_MODELOS / f).exists()}
    # Modelo propio del equipo (ACA 1, reentrenado en el notebook): archivo de confianza del repositorio
    rf = joblib.load(DIR_MODELOS / "random_forest_aca1.joblib")
    return esc, redes, rf


@st.cache_resource
def conexion():
    return db.conectar(RUTA_DB)


@st.cache_data
def tabla_clientes(_version_db: int) -> pd.DataFrame:
    ro = sqlite3.connect(f"file:{RUTA_DB.as_posix()}?mode=ro", uri=True)
    df = pd.read_sql_query("SELECT * FROM clientes ORDER BY CustomerID", ro)
    ro.close()
    return df.set_index("CustomerID")


met = cargar_metricas(_version(DIR_MODELOS / "metricas.json"))
esc, REDES, RF_ART = cargar_modelos(sum(_version(p) for p in DIR_MODELOS.glob("*.*")))
con = conexion()
CLIENTES = tabla_clientes(_version(RUTA_DB))
EJEMPLOS = pd.read_csv(DIR_DATA / "ejemplos_clientes.csv")
OPCIONES = sorted(REDES, key=lambda n: n != met["modelo_app"]) + [RF]


# ───────────────────────────── Funciones auxiliares ─────────────────────────────
def predecir(nombre: str, base: np.ndarray, rfm_exacto: pd.DataFrame | None = None) -> np.ndarray:
    """P(abandono) para una o varias secuencias base (n, T, 5). El RF usa R-F-M exactos si se conocen
    (cliente existente) y, si no, los derivados de la secuencia."""
    base = np.asarray(base, dtype=np.float32)
    if base.ndim == 2:
        base = base[None]
    if nombre != RF:
        return probabilidad(REDES[nombre], esc, base)
    if rfm_exacto is None:
        _, est = derivar(base)
        rfm_exacto = pd.DataFrame(est, columns=["recencia_dias", "frecuencia", "monetario"])
    return RF_ART["pipeline"].predict_proba(rfm_exacto[RF_ART["variables"]])[:, 1]


def decision(p: float, u: float) -> str:
    return CLASES[int(p >= u)]


def registrar(origen: str, customer_id, prob: float, clave_extra: str = "") -> None:
    """Guarda la predicción en SQLite una sola vez por entrada, modelo y umbral en la sesión."""
    clave = hashlib.md5(f"{origen}|{clave_extra}|{modelo_sel}|{umbral:.2f}".encode()).hexdigest()
    vistos = st.session_state.setdefault("registradas", set())
    if clave in vistos:
        return
    global con
    for _ in range(2):
        try:
            db.registrar_prediccion(con, origen, customer_id, modelo_sel, prob, umbral, decision(prob, umbral))
            vistos.add(clave)
            return
        except sqlite3.Error:
            # p. ej. tras un redespliegue que reemplazó ventas.db: se descarta la conexión y se reintenta
            conexion.clear()
            con = conexion()
    st.toast("No se pudo guardar esta predicción en el historial (la predicción sí es válida).", icon=":material/warning:")


def tabla_semanas(base: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(base, columns=BASE_CANALES)
    df.insert(0, "semana", np.arange(N_SEMANAS))
    df.insert(1, "inicio", [inicio_semana(t).date() for t in range(N_SEMANAS)])
    return df


def grafico_secuencia(base: np.ndarray, efectos: pd.DataFrame | None):
    """Gasto semanal; si hay explicación, cada barra se colorea por su efecto en el riesgo."""
    df = tabla_semanas(base)
    df["efecto"] = 0.0
    if efectos is not None and len(efectos):
        df.loc[efectos.semana, "efecto"] = efectos.efecto.fillna(0).to_numpy()
    df["lectura"] = np.where(df.pedidos == 0, "sin compra",
                             np.where(df.efecto > 0.005, "reduce el riesgo", np.where(df.efecto < -0.005, "aumenta el riesgo", "neutra")))
    lim = max(0.02, float(df.efecto.abs().max()))
    color = alt.Color("efecto:Q", title="Δ P(abandono) si se borra",
                      scale=alt.Scale(domain=[-lim, 0, lim], range=[COLORES["Abandono"], "#d9d9d9", COLORES["Recompra"]]))
    barras = alt.Chart(df).mark_bar().encode(
        x=alt.X("semana:O", title="Semana (39 = última antes del corte del 1-sep-2011)"),
        y=alt.Y("gasto:Q", title="Gasto semanal (£)"),
        color=color,
        tooltip=["semana", alt.Tooltip("inicio:T", title="inicio"), alt.Tooltip("gasto:Q", format=",.2f"), "pedidos",
                 "unidades", "productos", "devoluciones", alt.Tooltip("efecto:Q", format="+.3f"), "lectura"])
    return barras.properties(height=260)


def explicar(base: np.ndarray, p: float):
    """Texto y efectos por semana (oclusión) para el modelo seleccionado."""
    comp, est = derivar(base)
    r, f, m = est[0]
    if modelo_sel == RF:
        return None, (f"El Random Forest del ACA 1 sólo ve tres números: recencia ≈ {r:.0f} días, {f:.0f} pedidos y "
                      f"£{m:,.0f} de gasto total. No distingue el orden de las compras.")
    efectos, _ = oclusion_semanas(REDES[modelo_sel], esc, base)
    efectos = efectos.dropna()
    txt = [f"Última compra hace ≈ {r:.0f} días (semana {int(N_SEMANAS - 1 - r / 7)}), {f:.0f} pedidos y £{m:,.0f} en 40 semanas."]
    if len(efectos):
        top = efectos.reindex(efectos.efecto.abs().sort_values(ascending=False).index).head(3)
        partes = [f"semana {int(s)} ({'sin ella el riesgo subiría' if e > 0 else 'sin ella el riesgo bajaría'} {abs(e):.1%})"
                  for s, e in zip(top.semana, top.efecto)]
        txt.append("Semanas que más pesan en la decisión: " + "; ".join(partes) + ".")
    else:
        txt.append("Con una sola semana de compra la red no puede contrastar semanas: decide con la recencia y el monto.")
    return efectos, " ".join(txt)


def resultado(p: float):
    clase = decision(p, umbral)
    if clase == "Abandono":
        st.error(f"Riesgo de abandono: **{p:.1%}** (umbral {umbral:.2f}). Conviene una acción de retención "
                 "(contacto, cupón o recordatorio).", icon=":material/trending_down:")
    else:
        st.success(f"Probable recompra: P(abandono) = **{p:.1%}** (umbral {umbral:.2f}).", icon=":material/trending_up:")
    st.progress(min(max(p, 0.0), 1.0), text=f"P(abandono) = {p:.1%}")


# ───────────────────────────── Encabezado y barra lateral ───────────────────────
st.title(":material/shopping_cart: Abandono de clientes con Deep Learning")
st.caption("ACA N° 2 · Aprendizaje Automático · CUN – Especialización en Inteligencia Artificial · Redes recurrentes (GRU/LSTM) "
           "y Transformer temporal sobre la secuencia semanal de compras de cada cliente · UCI Online Retail")

with st.sidebar:
    st.header("Modelo")
    modelo_sel = st.radio(
        "Modelo", OPCIONES,
        captions=[("Recomendado: mejor ROC-AUC en validación" if n == met["modelo_app"] else
                   ("Línea base del ACA 1 (sólo R-F-M)" if n == RF else "Alternativa entrenada")) for n in OPCIONES])
    res = met["resultados_test"][modelo_sel]
    umbral = st.slider("Umbral de decisión", 0.05, 0.95, float(res["umbral"]), 0.01, key=f"umbral_{modelo_sel}",
                       help="Por defecto, el umbral elegido en validación (regla de Youden, la misma del ACA 1). "
                            "Bajarlo detecta más abandonos a costa de más falsas alarmas.")
    st.markdown(f"**Arquitectura:** {DESCRIPCION.get(modelo_sel, 'Random Forest con recencia, frecuencia y gasto')}  \n"
                f"**Clientes:** train {met['n']['train']} · val {met['n']['val']} · prueba {met['n']['test']}")
    st.divider()
    st.subheader("Desempeño en prueba")
    c1, c2 = st.columns(2)
    c1.metric("ROC-AUC", f"{res['roc_auc']:.3f}")
    c2.metric("F1", f"{res['f1']:.3f}")
    c1.metric("Recall", f"{res['recall']:.1%}")
    c2.metric("Precisión", f"{res['precision']:.1%}")

tabs = st.tabs([":material/person: Cliente existente", ":material/edit: Secuencia manual", ":material/upload_file: Lotes (CSV)",
                ":material/database: Historial (SQL)", ":material/monitoring: Desempeño", ":material/info: Acerca de"])

# ───────────────────────────── 1. Cliente existente ─────────────────────────────
with tabs[0]:
    ej_ids = EJEMPLOS.CustomerID.tolist()
    modo = st.segmented_control("Elegir cliente", ["Ejemplos de prueba", "Buscar por ID"], default="Ejemplos de prueba")
    if modo == "Buscar por ID":
        cid = st.selectbox("ID del cliente (3.314 clientes con compras antes del corte)", CLIENTES.index.tolist(),
                           format_func=lambda c: f"{c} · {CLIENTES.at[c, 'particion']} · {CLIENTES.at[c, 'segmento']}")
    else:
        cid = st.selectbox("Cliente de ejemplo (conjunto de prueba, nunca visto en el entrenamiento)", ej_ids,
                           format_func=lambda c: f"{c} · {CLIENTES.at[c, 'segmento']} · real: {CLASES[CLIENTES.at[c, 'abandono']]}")
    if cid is not None:
        info = CLIENTES.loc[cid]
        base = db.serie_cliente(con, cid, N_SEMANAS)
        p = float(predecir(modelo_sel, base, CLIENTES.loc[[cid], ["recencia_dias", "frecuencia", "monetario"]])[0])
        registrar(f"cliente:{cid}", cid, p)
        efectos, texto = explicar(base, p)
        a, b = st.columns([1, 1.6])
        with a:
            with st.container(border=True):
                st.markdown(f"**Cliente {cid}** · {info.pais} · segmento ACA 1: **{info.segmento}** · partición: `{info.particion}`")
                k1, k2, k3 = st.columns(3)
                k1.metric("Recencia (días)", int(info.recencia_dias))
                k2.metric("Pedidos", int(info.frecuencia))
                k3.metric("Gasto total", f"£{info.monetario:,.0f}")
            resultado(p)
            real = CLASES[int(info.abandono)]
            if info.particion == "test":
                st.caption(f"Resultado real tras el corte: **{real}** "
                           f"({'acierto' if decision(p, umbral) == real else 'error'} del modelo).")
            else:
                st.caption(f"Este cliente se usó en {'el entrenamiento' if info.particion == 'train' else 'la validación'}: "
                           f"su resultado real ({real}) no sirve para medir el acierto.")
            st.markdown("**Todos los modelos para este cliente**")
            comp = pd.DataFrame({"modelo": OPCIONES,
                                 "P(abandono)": [float(predecir(n, base, CLIENTES.loc[[cid], ["recencia_dias", "frecuencia", "monetario"]])[0])
                                                 for n in OPCIONES],
                                 "umbral": [met["resultados_test"][n]["umbral"] for n in OPCIONES]})
            comp["decisión"] = [decision(pp, uu) for pp, uu in zip(comp["P(abandono)"], comp.umbral)]
            st.dataframe(comp, hide_index=True, column_config={"P(abandono)": st.column_config.ProgressColumn(format="percent")})
        with b:
            st.markdown("**La «película» del cliente: gasto semana a semana**")
            st.altair_chart(grafico_secuencia(base, efectos))
            st.info(texto, icon=":material/lightbulb:")
            with st.expander("¿Cómo se calcula la explicación?"):
                st.markdown("Para cada semana con compra se borra esa semana y se vuelve a predecir (**oclusión**). Verde: la "
                            "compra de esa semana **reduce** el riesgo (sin ella sube). Naranja: la **aumenta**. Es la misma "
                            "técnica que el notebook promedia sobre los clientes de prueba.")

# ───────────────────────────── 2. Secuencia manual ──────────────────────────────
PLANTILLAS = {
    "Apagándose": {0: (250, 1), 4: (200, 1), 9: (150, 1), 15: (90, 1)},
    "Estable": {t: (150, 1) for t in range(1, 40, 3)},
    "Recuperándose": {3: (120, 1), 30: (200, 1), 34: (250, 1), 38: (300, 1)},
    "Un solo pedido": {12: (300, 1)},
}
with tabs[1]:
    st.subheader("Simula un cliente: escribe o ajusta sus compras semanales")
    plantilla = st.segmented_control("Plantilla de partida", list(PLANTILLAS), default="Apagándose", key="plantilla")
    plantilla = plantilla or "Apagándose"
    inicial = np.zeros((N_SEMANAS, len(BASE_CANALES)), dtype=np.float32)
    for t, (g, n) in PLANTILLAS[plantilla].items():
        inicial[t] = [g, n, round(g / 3), max(1, round(g / 25)), 0]
    editada = st.data_editor(tabla_semanas(inicial), key=f"editor_{plantilla}", hide_index=True, height=300,
                             disabled=["semana", "inicio"],
                             column_config={"gasto": st.column_config.NumberColumn("gasto (£)", min_value=0.0, format="%.2f"),
                                            **{c: st.column_config.NumberColumn(c, min_value=0, step=1)
                                               for c in BASE_CANALES[1:]}})
    base_m = secuencia_desde_tabla(editada)
    if not (base_m[:, 1] > 0).any():
        st.warning("La secuencia necesita al menos una semana con pedidos > 0.")
    else:
        p = float(predecir(modelo_sel, base_m)[0])
        registrar(f"manual:{plantilla}", None, p, hashlib.md5(base_m.tobytes()).hexdigest())
        a, b = st.columns([1, 1.6])
        with a:
            resultado(p)
            efectos, texto = explicar(base_m, p)
            st.info(texto, icon=":material/lightbulb:")
        with b:
            st.altair_chart(grafico_secuencia(base_m, efectos))
        st.caption("Días sin comprar y R-F-M se derivan de la secuencia con la misma función que usó el entrenamiento "
                   "(`utils/secuencias.derivar`). En secuencias manuales la recencia se mide en semanas completas.")

# ───────────────────────────── 3. Lotes ─────────────────────────────────────────
with tabs[2]:
    st.subheader("Predicción por lotes")
    st.markdown("Sube un CSV en **formato largo** (una fila por cliente y semana con compra: `cliente, semana, gasto, pedidos, "
                "unidades, productos, devoluciones`, con `semana` de 0 a 39) **o** un CSV con una sola columna `CustomerID` "
                "de clientes existentes.")
    archivo = st.file_uploader("Archivo CSV", type=["csv"])
    usar_ej = st.checkbox("Usar el lote de ejemplo (20 clientes del conjunto de prueba)")
    st.download_button("Descargar plantilla (lote de ejemplo)", (DIR_DATA / "lote_ejemplo.csv").read_bytes(),
                       "lote_ejemplo.csv", "text/csv", icon=":material/download:")
    df_in, nombre_lote = None, None
    if archivo is not None:
        df_in, nombre_lote = pd.read_csv(archivo), archivo.name
    elif usar_ej:
        df_in, nombre_lote = pd.read_csv(DIR_DATA / "lote_ejemplo.csv"), "lote_ejemplo.csv"
    if df_in is not None:
        try:
            cols = {c.lower(): c for c in df_in.columns}
            if set(cols) == {"customerid"}:
                ids = [int(c) for c in df_in[cols["customerid"]] if int(c) in CLIENTES.index]
                faltan = len(df_in) - len(ids)
                bases = np.stack([db.serie_cliente(con, c, N_SEMANAS) for c in ids])
                rfm = CLIENTES.loc[ids, ["recencia_dias", "frecuencia", "monetario"]].reset_index(drop=True)
                if faltan:
                    st.warning(f"{faltan} ID(s) no existen en la base y se omiten.")
            else:
                df_in = df_in.rename(columns=str.lower)
                falt = [c for c in ["cliente", "semana"] + BASE_CANALES if c not in df_in.columns]
                if falt:
                    raise ValueError(f"faltan columnas: {', '.join(falt)}")
                if not df_in.semana.between(0, N_SEMANAS - 1).all():
                    raise ValueError(f"la columna semana debe estar entre 0 y {N_SEMANAS - 1}")
                ids = sorted(df_in.cliente.unique())
                bases = np.zeros((len(ids), N_SEMANAS, len(BASE_CANALES)), dtype=np.float32)
                pos = {c: i for i, c in enumerate(ids)}
                agr = df_in.groupby(["cliente", "semana"])[BASE_CANALES].sum().reset_index()
                bases[agr.cliente.map(pos).to_numpy(), agr.semana.to_numpy().astype(int)] = agr[BASE_CANALES].clip(lower=0).to_numpy()
                rfm = None
            validos = (bases[:, :, 1] > 0).any(1)
            probs = np.full(len(ids), np.nan)
            if validos.any():
                probs[validos] = predecir(modelo_sel, bases[validos], None if rfm is None else rfm[validos])
            _, est = derivar(bases)
            out = pd.DataFrame({"cliente": ids, "pedidos": est[:, 1].astype(int), "gasto_total": est[:, 2].round(2),
                                "prob_abandono": probs})
            out["decision"] = [decision(p_, umbral) if not np.isnan(p_) else "sin pedidos" for p_ in probs]
            for c, p_ in zip(ids, probs):
                if not np.isnan(p_):
                    registrar(f"lote:{nombre_lote}", c if isinstance(c, (int, np.integer)) else None, float(p_), str(c))
            k1, k2, k3 = st.columns(3)
            k1.metric("Clientes", len(out))
            k2.metric("En riesgo de abandono", int((out.decision == "Abandono").sum()))
            k3.metric("Probable recompra", int((out.decision == "Recompra").sum()))
            st.dataframe(out.sort_values("prob_abandono", ascending=False), hide_index=True,
                         column_config={"prob_abandono": st.column_config.ProgressColumn("P(abandono)", format="percent"),
                                        "gasto_total": st.column_config.NumberColumn("gasto total (£)", format="%.2f")})
            st.download_button("Descargar resultados (CSV)", out.to_csv(index=False).encode("utf-8"),
                               "predicciones_abandono.csv", "text/csv", icon=":material/download:")
        except Exception as e:  # noqa: BLE001
            st.error(f"No se pudo procesar el archivo: {e}")

# ───────────────────────────── 4. Historial SQL ─────────────────────────────────
with tabs[3]:
    st.subheader("Base de datos SQLite (`data/ventas.db`)")
    try:
        hist = db.consultar(con, "SELECT * FROM predicciones ORDER BY id DESC")
    except sqlite3.Error:
        conexion.clear(); con = conexion()
        hist = db.consultar(con, "SELECT * FROM predicciones ORDER BY id DESC")
    a1, a2 = st.columns([1, 2.2])
    with a1:
        st.metric("Predicciones registradas", len(hist))
        if len(hist):
            cnt = hist.decision.value_counts().reindex(CLASES, fill_value=0).rename_axis("decisión").reset_index(name="n")
            st.altair_chart(alt.Chart(cnt).mark_bar().encode(
                x=alt.X("decisión:N", title=None), y=alt.Y("n:Q", title="predicciones"),
                color=alt.Color("decisión:N", scale=alt.Scale(domain=CLASES, range=[COLORES[c] for c in CLASES]), legend=None)
            ).properties(height=220))
    with a2:
        st.markdown("**Últimas predicciones** (tabla `predicciones`)")
        st.dataframe(hist.head(200), hide_index=True, height=280)

    st.markdown("**Consultas SQL** (tablas `transacciones`, `productos`, `clientes`, `series`, `predicciones`, "
                "`predicciones_prueba`; vistas `ventas_limpias`, `cancelaciones`)")
    CONSULTAS = {
        "Acierto en prueba por segmento y modelo": f"""SELECT c.segmento, p.modelo, COUNT(*) AS clientes,
       ROUND(100.0 * AVG(p.prediccion = p.real), 1) AS pct_acierto,
       ROUND(100.0 * SUM(p.prediccion = 1 AND p.real = 1) / SUM(p.real = 1), 1) AS recall
FROM predicciones_prueba p JOIN clientes c USING (CustomerID)
GROUP BY c.segmento, p.modelo ORDER BY c.segmento, pct_acierto DESC""",
        "Clientes por partición y tasa de abandono": """SELECT particion, COUNT(*) AS clientes, ROUND(100.0 * AVG(abandono), 1) AS pct_abandono
FROM clientes GROUP BY particion""",
        "Actividad semanal: abandono vs. recompra": """SELECT s.semana, c.abandono, COUNT(*) AS clientes_que_compran, ROUND(SUM(s.gasto)) AS gasto
FROM series s JOIN clientes c USING (CustomerID)
GROUP BY s.semana, c.abandono ORDER BY s.semana, c.abandono""",
        "Ventas mensuales por país (top 5)": """SELECT strftime('%Y-%m', InvoiceDate) AS mes, Country AS pais, ROUND(SUM(Quantity * UnitPrice)) AS ventas
FROM ventas_limpias
WHERE Country IN (SELECT Country FROM ventas_limpias GROUP BY Country ORDER BY SUM(Quantity * UnitPrice) DESC LIMIT 5)
GROUP BY mes, pais ORDER BY mes, ventas DESC""",
        "Productos más vendidos": """SELECT p.Description AS producto, SUM(v.Quantity) AS unidades, COUNT(DISTINCT v.CustomerID) AS clientes
FROM ventas_limpias v JOIN productos p USING (StockCode)
GROUP BY v.StockCode ORDER BY unidades DESC LIMIT 15""",
        "Historial de la app: alto riesgo": """SELECT fecha, origen, modelo, ROUND(prob_abandono, 3) AS prob, decision
FROM predicciones WHERE prob_abandono >= 0.5 ORDER BY prob_abandono DESC""",
    }
    sel = st.selectbox("Consulta predefinida", list(CONSULTAS))
    sql = st.text_area("SQL (sólo lectura: se permite únicamente SELECT)", CONSULTAS[sel], height=140, key=f"sql_{sel}")
    if st.button("Ejecutar consulta", icon=":material/play_arrow:"):
        if not sql.strip().lower().startswith(("select", "with")):
            st.error("Sólo se permiten consultas SELECT.")
        else:
            try:
                ro = sqlite3.connect(f"file:{RUTA_DB.as_posix()}?mode=ro", uri=True)
                st.dataframe(pd.read_sql_query(sql, ro), hide_index=True)
                ro.close()
            except Exception as e:  # noqa: BLE001
                st.error(f"Error en la consulta: {e}")

# ───────────────────────────── 5. Desempeño ─────────────────────────────────────
with tabs[4]:
    st.subheader(f"Evaluación sobre {met['n']['test']} clientes de prueba (los mismos del ACA 1)")
    ic = res["ic95"]
    cols = st.columns(4)
    for col, (k, nombre) in zip(cols, [("roc_auc", "ROC-AUC"), ("f1", "F1"), ("recall", "Recall"), ("precision", "Precisión")]):
        col.metric(nombre, f"{res[k]:.3f}", help=f"IC 95 % (bootstrap): {ic[k][0]:.3f} – {ic[k][1]:.3f}", border=True)
    st.markdown("**Comparación de modelos** (el modelo de la app se eligió por ROC-AUC en **validación**)")
    filas = []
    for n, r in met["resultados_test"].items():
        d = met["vs_rf"].get(n)
        filas.append({"modelo": n, "parámetros": r["parametros"], "AUC val": met["seleccion_val"].get(n, {}).get("auc_val"),
                      "umbral": r["umbral"], "accuracy": r["accuracy"], "ROC-AUC": r["roc_auc"], "PR-AUC": r["pr_auc"], "F1": r["f1"],
                      "recall": r["recall"], "precisión": r["precision"],
                      "ΔAUC vs RF [IC 95 %]": f"{d['delta_auc']:+.3f} [{d['ic95'][0]:+.3f}, {d['ic95'][1]:+.3f}]" if d else "—"})
    st.dataframe(pd.DataFrame(filas), hide_index=True,
                 column_config={k: st.column_config.NumberColumn(format="%.3f")
                                for k in ("AUC val", "accuracy", "ROC-AUC", "PR-AUC", "F1", "recall", "precisión")})
    for archivo, titulo in [("05_curvas_entrenamiento.png", "Curvas de pérdida, accuracy y ROC-AUC por época (entrenamiento y validación)"),
                            ("06_roc_pr.png", "Curvas ROC y Precisión–Recall"), ("07_matrices_confusion.png", "Matrices de confusión"),
                            ("09_desglose.png", "¿Dónde aporta la secuencia? ROC-AUC por segmento y por número de pedidos"),
                            ("10_interpretabilidad.png", "Interpretabilidad: permutación, oclusión por semana y atención"),
                            ("11_arquitectura_" + met["modelo_app"].lower() + ".png", f"Arquitectura del modelo de la app ({met['modelo_app']})")]:
        ruta = DIR_FIGURAS / archivo
        if ruta.exists():
            st.markdown(f"**{titulo}**")
            st.image(str(ruta), alt=titulo)

# ───────────────────────────── 6. Acerca de ─────────────────────────────────────
with tabs[5]:
    n = met["n"]
    st.markdown(f"""
### Flujo metodológico
1. **Exploración y gestión SQL:** {n['lineas_originales']:,} líneas de factura de *Online Retail* cargadas en SQLite; calidad, EDA y limpieza con consultas y vistas SQL ({n['lineas_limpias']:,} líneas de venta limpias).
2. **Preparación y pipeline:** {n['clientes']:,} clientes, cada uno con una secuencia de {N_SEMANAS} semanas × 7 canales hasta el corte del 1-sep-2011; misma etiqueta y misma partición de prueba que el ACA 1; escalado ajustado sólo en entrenamiento; aumento de datos (escalado, jitter, recorte de ventana); `pos_weight` para el desbalance.
3. **Modelado neuronal:** GRU, LSTM y Transformer temporal (ensambles de {len(met['semillas'])} semillas) frente al Random Forest R-F-M del ACA 1. La app usa **{met['modelo_app']}**, el de mayor ROC-AUC en validación.
4. **Evaluación:** pérdida y ROC-AUC por época, ROC/PR, matrices de confusión, IC 95 % bootstrap, diferencia pareada contra el ACA 1, desglose por segmento e interpretabilidad (permutación, oclusión, atención).
5. **Despliegue:** esta aplicación (inferencia en CPU, sin internet ni dataset crudo; historial en SQLite).

### Limitaciones
- Un solo comercio y un año de datos; la ventana de etiqueta (sep–dic) cae en temporada alta.
- Secuencias dispersas: {n['un_solo_pedido']:.0%} de los clientes compró una sola vez antes del corte.
- Herramienta educativa; en Streamlit Cloud el historial SQLite se reinicia al redesplegar.

### Referencias
- Chen, D. (2015). *Online Retail* [Conjunto de datos]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33
- Cho, K., et al. (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. *Proceedings of EMNLP 2014*, 1724–1734.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735–1780.
- Vaswani, A., et al. (2017). Attention is all you need. *Advances in Neural Information Processing Systems 30*, 5998–6008.
""")
