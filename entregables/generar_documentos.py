"""Genera README.md y el guion del video a partir de modelos/metricas.json (ninguna cifra se escribe a mano).

Uso:  python entregables/generar_documentos.py
Los enlaces se leen de entregables/enlace_*.txt (los guarda generar_poster.py).
"""
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "entregables"
m = json.loads((BASE / "modelos" / "metricas.json").read_text(encoding="utf-8"))
CONCL = json.loads((OUT / "conclusiones.json").read_text(encoding="utf-8"))
R, APP, RF = m["resultados_test"], m["modelo_app"], "Random Forest (ACA 1)"
app, rf, d = R[APP], R[RF], m["vs_rf"][APP]
pol_a, pol_r = m["politica_recall80"][APP], m["politica_recall80"][RF]
n = m["n"]


def leer(k, defecto):
    f = OUT / f"enlace_{k}.txt"
    return f.read_text(encoding="utf-8").strip() if f.exists() and f.read_text(encoding="utf-8").strip() else defecto


URL_APP, URL_VIDEO, URL_REPO = leer("app", "(por publicar)"), leer("video", "(por publicar)"), leer("repo", "(por publicar)")
f3 = lambda x: f"{x:.3f}".replace(".", ",")
s3 = lambda x: ("−" if math.copysign(1, x) < 0 else "+") + f3(abs(x))   # conserva el signo de −0,0000x
pc = lambda x, dec=1: f"{100 * x:.{dec}f}".replace(".", ",") + " %"
mil = lambda x: f"{x:,}".replace(",", ".")
# Para la voz: números en palabras legibles ("cero coma siete tres uno" se lee mal) → se usan decimales con coma
v3 = lambda x: f"{x:.2f}".replace(".", ",")

# ─────────────────────────────── README ───────────────────────────────
filas = []
for k, v in R.items():
    dd = m["vs_rf"].get(k)
    nombre = f"**{k}** (modelo de la app)" if k == APP else k
    dif = f"{s3(dd['delta_auc'])} [{s3(dd['ic95'][0])}; {s3(dd['ic95'][1])}]" if dd else "—"
    filas.append(f"| {nombre} | {mil(v['parametros']) if v.get('parametros') else '—'} | "
                 f"{f3(m['seleccion_val'][k]['auc_val']) if k in m['seleccion_val'] else '—'} | {f3(v['accuracy'])} | {f3(v['roc_auc'])} | "
                 f"{f3(v['pr_auc'])} | {f3(v['f1'])} | {f3(v['recall'])} | {f3(v['precision'])} | {dif} |")
imp = m["importancia"]
readme = f"""# ACA N° 2 – Aprendizaje Automático (CUN) · Pilar Ventas
**Del ML clásico al Deep Learning:** predicción del abandono de clientes a partir de su **secuencia semanal de compras** con redes recurrentes (GRU, LSTM) y un **Transformer temporal**, comparadas con el Random Forest del ACA 1 y desplegadas en Streamlit.

**Integrantes:**
- Juan Guillermo Angel Jaramillo
- Luis Bernardo Correa Restrepo
- María Alejandra Manga Betancourt

**Docente:** Hamilton Rivera Flor · **Dataset:** UCI Online Retail (Chen, 2015) · **Continuación de:** ACA 1 (segmentación RFM + Random Forest)

**App en línea:** {URL_APP} · **Video:** {URL_VIDEO}

## Resultados (conjunto de prueba: los mismos {n['test']} clientes del ACA 1)
| Modelo | Parámetros | ROC-AUC val | Accuracy | ROC-AUC | PR-AUC | F1 | Recall | Precisión | ΔAUC vs RF [IC 95 %] |
|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(filas)}

- El modelo de la app se eligió por el mayor **ROC-AUC en validación**; la prueba sólo se usa para reportar. Cada red es un ensamble de {len(m['semillas'])} semillas.
- {CONCL[1]}
- **Mismo punto de operación** (recall ≥ 0,80, umbral fijado en validación): {APP} detecta {pol_a['VP']} abandonos con {pol_a['FP']} falsas alarmas (precisión {f3(pol_a['precision'])}); el RF, {pol_r['VP']} con {pol_r['FP']} (precisión {f3(pol_r['precision'])}).
- Interpretabilidad ({APP}): lo más influyente es **{imp[0]['variable']}**; borrar una compra de las últimas 8 semanas sube P(abandono) en {s3(m['oclusion_ultimas8'])} frente a {s3(m['oclusion_primer_trimestre'])} en el primer trimestre.
- La app acierta {m['ejemplos']['aciertos']} de sus {m['ejemplos']['n']} ejemplos (clientes de prueba elegidos al azar por segmento y clase, sin mirar la predicción).
- Pérdida inicial verificada antes de entrenar: {f3(m['perdida_inicial'][APP])} (esperada {f3(m['perdida_inicial_esperada'])}). El notebook completo tarda ≈ {m['minutos_notebook']:.0f} min en CPU.

## Contenido
| Ruta | Qué es |
|---|---|
| `ACA2_Deep_Learning_Ventas.ipynb` | Notebook completo: SQL → EDA → limpieza → secuencias y aumento → 3 redes + RF → evaluación → interpretabilidad → exportación |
| `app.py` | App Streamlit: cliente existente (con explicación por semana), secuencia manual, lotes CSV, historial y consultas SQL, desempeño, selector de modelos |
| `utils/` | `db.py` (SQLite), `secuencias.py` (secuencias, escalado, aumento), `modelos.py` (GRU/LSTM/Transformer, ensamble), `entrenamiento.py`, `inferencia.py`, `diagrama.py` |
| `data/ventas.db` | SQLite: `transacciones`, `productos`, `clientes` (con partición), `series`, `predicciones` (historial de la app), `predicciones_prueba`; vistas de limpieza |
| `data/clientes_aca1.csv`, `aca1/` | Tabla de clientes y modelo del ACA 1 (verificación de la misma población, etiqueta y línea base) |
| `data/ejemplos_clientes.csv`, `data/lote_ejemplo.csv` | Ejemplos de la app y lote CSV de ejemplo (clientes de **prueba**) |
| `modelos/` | Pesos de las 3 redes (5 semillas c/u), escalador, RF y `metricas.json` |
| `figuras/` | Gráficos del notebook, diagramas de arquitectura y vista previa de la app |
| `entregables/` | Póster (`Poster_ACA2_Ventas_DL.pdf`, `poster.html`, `generar_poster.py`), guion del video y `generar_documentos.py` |

## Ejecutar la app (Windows, PowerShell)
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/streamlit run app.py
```
La app no necesita internet ni el dataset crudo: usa `modelos/` y `data/ventas.db`.

## Reentrenar (opcional, ≈ {m['minutos_notebook']:.0f} min en CPU)
```bash
.venv/Scripts/python -m pip install -r requirements-notebook.txt
.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace ACA2_Deep_Learning_Ventas.ipynb --ExecutePreprocessor.timeout=10800
.venv/Scripts/python entregables/generar_poster.py
.venv/Scripts/python entregables/generar_documentos.py
```
Si `data/raw/onlineretail.rda` no existe, el notebook lo descarga de la copia pública del paquete R `onlineretail`. Con `ACA2_RAPIDO=1` corre una prueba de 1 época por modelo que escribe en `modelos/_rapido/` y `figuras/_rapido/`.

## Decisiones técnicas clave
- **Misma etiqueta y misma partición que el ACA 1:** el notebook verifica con `assert` que los {mil(n['clientes'])} clientes, la etiqueta, la recencia y la frecuencia coinciden, y que el RF reproduce su ROC-AUC ({f3(rf['roc_auc'])}).
- **Sin fuga:** X usa sólo transacciones anteriores al 1-sep-2011 y la etiqueta sólo posteriores. Cada cliente está en una sola partición. El escalado se ajusta sólo con entrenamiento.
- **Un solo preprocesamiento:** `utils/secuencias.derivar` calcula días sin comprar y R-F-M para el notebook, la app, el editor manual y los CSV. El notebook comprueba con `torch.equal` que los tensores de la app son idénticos.
- **Aumento para series:** escalado aleatorio, jitter sólo en semanas activas y recorte de ventana, aplicados antes de derivar para que la secuencia siga siendo coherente.
- **Diferencia en un cliente:** el `DISTINCT` de SQL ignora la descripción del producto. En el cliente 12909 dos líneas que sólo cambiaban de descripción cuentan como duplicado (diferencia de £14,85 en el gasto). El RF se evalúa con la tabla original del ACA 1.

## Desplegar en Streamlit Community Cloud
1. Subir esta carpeta a GitHub (`.gitignore` excluye `.venv/` y `data/raw/`).
2. En https://share.streamlit.io → **Create app** → *Deploy a public app from GitHub* → repositorio, rama `main`, archivo `app.py`; en *Advanced settings*, Python 3.12.
3. El historial SQLite se reinicia en cada redespliegue (limitación de la nube gratuita).
"""
(BASE / "README.md").write_text(readme, encoding="utf-8")

# ─────────────────────────────── Guion ───────────────────────────────
JUAN, LUIS, MARIA = "Juan Guillermo Angel", "Luis Bernardo Correa", "María Alejandra Manga"
seg = [x for x in m["desglose"] if x["desglose"] == "segmento"]
g1 = max(seg, key=lambda x: x[APP] - x[RF])
escenas = [
    ("1 · 0:00–0:30 (30 s)", "Encabezado del póster (título, integrantes).", JUAN,
     "Cordial saludo. Presentamos el ACA número 2 de Aprendizaje Automático de la Especialización en Inteligencia Artificial de la CUN, "
     "con el docente Hamilton Rivera Flor. Somos Juan Guillermo Angel, Luis Bernardo Correa y María Alejandra Manga. En el ACA 1 "
     "predijimos qué clientes de una tienda en línea dejarían de comprar, con un Random Forest que veía sólo tres números por cliente: "
     "recencia, frecuencia y gasto. Es como mirar una foto. En el ACA 2 la red neuronal ve la película completa: la historia de compras semana a semana."),
    ("2 · 0:30–1:15 (45 s)", "`figuras/01_ventas_mensuales_pais.png` → en la app, pestaña **Historial (SQL)**, consulta «Ventas mensuales por país» → "
     "`figuras/03_mapa_clientes_semanas.png`.", LUIS,
     f"Usamos el dataset Online Retail de UCI: {mil(n['lineas_originales'])} líneas de factura entre diciembre de 2010 y diciembre de 2011. "
     "Todo se gestiona en una base de datos SQLite: la calidad, el análisis exploratorio y la limpieza se hacen con consultas y vistas SQL. "
     "Quitamos líneas sin cliente, duplicados, cancelaciones y precios no válidos. Después convertimos a cada uno de los "
     f"{mil(n['clientes'])} clientes en una secuencia de 40 semanas con siete canales: gasto, pedidos, unidades, productos, devoluciones, "
     "días sin comprar y si ya era cliente. En este mapa se ve la diferencia: quienes abandonan se van apagando antes del corte."),
    ("3 · 1:15–2:05 (50 s)", "`figuras/04_aumento_datos.png` → diagrama de arquitectura del póster.", MARIA,
     "Para que la comparación sea justa usamos la misma etiqueta y exactamente los mismos clientes de prueba del ACA 1, y lo verificamos "
     "en el código. Nada del futuro entra en las secuencias y cada cliente está en una sola partición. En entrenamiento aplicamos aumento "
     "de datos para series: escalado, ruido y recorte de la ventana. Comparamos tres redes: una GRU y una LSTM, que leen las semanas en "
     "orden y deciden qué recordar, y un Transformer temporal, en el que cada semana es un token y la autoatención relaciona todas las "
     f"semanas entre sí. Cada red se entrenó con cinco semillas y elegimos la de mejor resultado en validación: la {APP}."),
    ("4 · 2:05–2:55 (50 s)", "Bloque de resultados del póster (KPIs y tabla) → `figuras/13_poster_modelo_app.png` → `figuras/09_desglose.png`.", JUAN,
     f"En los {n['test']} clientes de prueba la {APP} alcanza un área bajo la curva ROC de {v3(app['roc_auc'])}, frente a {v3(rf['roc_auc'])} "
     "del Random Forest. Las tres redes superan al modelo clásico por un margen parecido, pero el intervalo de confianza de la diferencia "
     "toca el cero, así que la mejora es pequeña y no concluyente: con compras tan esporádicas, buena parte de la señal ya estaba en "
     f"recencia, frecuencia y gasto. Con la misma meta de detectar al menos el 80 por ciento de los abandonos, la red da {pol_a['FP']} falsas "
     f"alarmas frente a {pol_r['FP']} del Random Forest. Donde más aporta la secuencia es en el segmento {g1['grupo']}."
     if d["ic95"][0] <= 0 else
     f"En los {n['test']} clientes de prueba la {APP} alcanza un área bajo la curva ROC de {v3(app['roc_auc'])}, frente a {v3(rf['roc_auc'])} "
     "del Random Forest, y la diferencia es estadísticamente significativa. "
     f"Con la misma meta de detectar al menos el 80 por ciento de los abandonos, la red da {pol_a['FP']} falsas alarmas frente a "
     f"{pol_r['FP']} del Random Forest. Donde más aporta la secuencia es en el segmento {g1['grupo']}."),
    ("5 · 2:55–4:00 (65 s)", "**Demo en vivo**: 1) pestaña *Cliente existente* con un ejemplo de prueba (probabilidad, gráfico semanal coloreado "
     "y explicación); 2) cambiar de modelo en la barra lateral; 3) *Secuencia manual* con las plantillas «Apagándose» y «Estable»; "
     "4) *Lotes (CSV)* con el lote de ejemplo y descarga; 5) *Historial (SQL)*.", LUIS,
     "Esta es la aplicación en Streamlit. Elegimos un cliente del conjunto de prueba: la red calcula su probabilidad de abandono y colorea "
     "cada semana según cuánto pesó en la decisión. Para eso borramos una semana a la vez y medimos cuánto cambia la predicción. Podemos "
     "cambiar de modelo y comparar con el Random Forest del ACA 1. En la pestaña de secuencia manual simulamos un cliente que se va "
     "apagando y otro estable, y vemos cómo cambia el riesgo. También procesamos un archivo CSV por lotes y descargamos el resultado. "
     "Cada predicción queda guardada en la base SQLite y se puede consultar con SQL desde la misma app."),
    ("6 · 4:00–4:30 (30 s)", "Conclusiones del póster → enlaces de App, Video y Código.", MARIA,
     "En conclusión, las redes recurrentes y de atención aprenden el ritmo de compra y la importancia de lo reciente, e igualan o superan "
     "ligeramente al Random Forest en el mismo problema. Su límite son los clientes con una sola compra. Lo valioso para el negocio es "
     "una herramienta que explica cada predicción semana a semana. El código, el notebook y la app están en los enlaces del póster. Muchas gracias."),
]
palabras = sum(len(e[3].split()) for e in escenas)
tabla = "\n".join(f"| **{t}** | {v} | {who} | \"{txt}\" |" for t, v, who, txt in escenas)
continuo = "\n>\n".join(f"> {e[3]}" for e in escenas)
guion = f"""# GUION TÉCNICO Y LOCUCIÓN DE VIDEO (3 A 5 MINUTOS)
## PROYECTO ACA N° 2 – APRENDIZAJE AUTOMÁTICO · PILAR VENTAS
**Tema:** Del ML clásico al Deep Learning: predicción del abandono de clientes a partir de su secuencia semanal de compras (GRU, LSTM y Transformer) y despliegue en Streamlit
**Institución:** Corporación Unificada Nacional de Educación Superior (CUN) · Especialización en Inteligencia Artificial
**Docente:** Hamilton Rivera Flor
**Integrantes:** Juan Guillermo Angel Jaramillo · Luis Bernardo Correa Restrepo · María Alejandra Manga Betancourt

**Duración estimada:** ≈ {palabras / 135:.1f} min ({palabras} palabras a ≈ 135 palabras/min).
**Enlaces para la demo:**
- App: {URL_APP}
- Código: {URL_REPO}
- Póster: `entregables/Poster_ACA2_Ventas_DL.pdf`

> **Antes de grabar:** abre la app unos minutos antes (en Streamlit Cloud tarda ≈ 30 s en «despertar» si estuvo inactiva) y deja listo el póster en PDF.

---

## TABLA ESCENA POR ESCENA

| Escena y tiempo | Recurso visual en pantalla | Locutor sugerido | Texto de locución |
|---|---|---|---|
{tabla}

---

## TEXTO CONTINUO PARA VOZ CON IA (TTS)
*(Para ElevenLabs, Clipchamp, Fliki, HeyGen, etc. Copia este bloque tal cual.)*

{continuo}

---

## LISTA DE CONTROL ANTES DE ENTREGAR
- [ ] Duración entre 3:00 y 5:00.
- [ ] Se ve la app en vivo y el póster.
- [ ] Subir el video (YouTube «no listado» o Google Drive «cualquiera con el enlace») y regenerar póster y documentos:
  `.venv/Scripts/python entregables/generar_poster.py --video https://TU-VIDEO` y `.venv/Scripts/python entregables/generar_documentos.py`
"""
(OUT / "Guion_Video_3-5min.md").write_text(guion, encoding="utf-8")
print(f"README.md y Guion_Video_3-5min.md generados ({palabras} palabras de locución)")
