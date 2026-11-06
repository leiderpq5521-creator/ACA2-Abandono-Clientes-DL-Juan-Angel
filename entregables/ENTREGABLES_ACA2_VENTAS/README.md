# ACA N° 2 – Aprendizaje Automático (CUN) · Pilar Ventas
**Del ML clásico al Deep Learning:** predicción del abandono de clientes a partir de su **secuencia semanal de compras** con redes recurrentes (GRU, LSTM) y un **Transformer temporal**, comparadas con el Random Forest del ACA 1 y desplegadas en Streamlit.

**Integrantes:**
- Juan Guillermo Angel Jaramillo
- Luis Bernardo Correa Restrepo
- María Alejandra Manga Betancourt

**Docente:** Hamilton Rivera Flor · **Dataset:** UCI Online Retail (Chen, 2015) · **Continuación de:** ACA 1 (segmentación RFM + Random Forest)

**App en línea:** https://aca2-abandono-clientes-dl-juan-angel-tuqs2bahdtmtheu3bhrzfa.streamlit.app/ · **Video:** (por publicar)

## Resultados (conjunto de prueba: los mismos 663 clientes del ACA 1)
| Modelo | Parámetros | ROC-AUC val | Accuracy | ROC-AUC | PR-AUC | F1 | Recall | Precisión | ΔAUC vs RF [IC 95 %] |
|---|---|---|---|---|---|---|---|---|---|
| Random Forest (ACA 1) | — | — | 0,642 | 0,720 | 0,590 | 0,654 | 0,821 | 0,544 | — |
| **GRU** (modelo de la app) | 11.457 | 0,774 | 0,662 | 0,731 | 0,599 | 0,622 | 0,674 | 0,577 | +0,011 [−0,000; +0,022] |
| LSTM | 14.881 | 0,771 | 0,658 | 0,730 | 0,598 | 0,640 | 0,740 | 0,564 | +0,010 [−0,002; +0,021] |
| Transformer | 19.937 | 0,769 | 0,661 | 0,731 | 0,600 | 0,648 | 0,758 | 0,566 | +0,011 [−0,001; +0,024] |

- El modelo de la app se eligió por el mayor **ROC-AUC en validación**; la prueba sólo se usa para reportar. Cada red es un ensamble de 5 semillas.
- Las tres redes superan al RF por un margen similar, pero el intervalo de la diferencia toca el cero: la mejora es consistente aunque pequeña y no concluyente con 663 clientes. Con secuencias tan dispersas, buena parte de la señal ya está en R-F-M.
- **Mismo punto de operación** (recall ≥ 0,80, umbral fijado en validación): GRU detecta 218 abandonos con 170 falsas alarmas (precisión 0,562); el RF, 224 con 188 (precisión 0,544).
- Interpretabilidad (GRU): lo más influyente es **Productos distintos**; borrar una compra de las últimas 8 semanas sube P(abandono) en +0,113 frente a +0,041 en el primer trimestre.
- La app acierta 6 de sus 8 ejemplos (clientes de prueba elegidos al azar por segmento y clase, sin mirar la predicción).
- Pérdida inicial verificada antes de entrenar: 0,814 (esperada 0,814). El notebook completo tarda ≈ 19 min en CPU.

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

## Reentrenar (opcional, ≈ 19 min en CPU)
```bash
.venv/Scripts/python -m pip install -r requirements-notebook.txt
.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace ACA2_Deep_Learning_Ventas.ipynb --ExecutePreprocessor.timeout=10800
.venv/Scripts/python entregables/generar_poster.py
.venv/Scripts/python entregables/generar_documentos.py
```
Si `data/raw/onlineretail.rda` no existe, el notebook lo descarga de la copia pública del paquete R `onlineretail`. Con `ACA2_RAPIDO=1` corre una prueba de 1 época por modelo que escribe en `modelos/_rapido/` y `figuras/_rapido/`.

## Decisiones técnicas clave
- **Misma etiqueta y misma partición que el ACA 1:** el notebook verifica con `assert` que los 3.314 clientes, la etiqueta, la recencia y la frecuencia coinciden, y que el RF reproduce su ROC-AUC (0,720).
- **Sin fuga:** X usa sólo transacciones anteriores al 1-sep-2011 y la etiqueta sólo posteriores. Cada cliente está en una sola partición. El escalado se ajusta sólo con entrenamiento.
- **Un solo preprocesamiento:** `utils/secuencias.derivar` calcula días sin comprar y R-F-M para el notebook, la app, el editor manual y los CSV. El notebook comprueba con `torch.equal` que los tensores de la app son idénticos.
- **Aumento para series:** escalado aleatorio, jitter sólo en semanas activas y recorte de ventana, aplicados antes de derivar para que la secuencia siga siendo coherente.
- **Diferencia en un cliente:** el `DISTINCT` de SQL ignora la descripción del producto. En el cliente 12909 dos líneas que sólo cambiaban de descripción cuentan como duplicado (diferencia de £14,85 en el gasto). El RF se evalúa con la tabla original del ACA 1.

## Desplegar en Streamlit Community Cloud
1. Subir esta carpeta a GitHub (`.gitignore` excluye `.venv/` y `data/raw/`).
2. En https://share.streamlit.io → **Create app** → *Deploy a public app from GitHub* → repositorio, rama `main`, archivo `app.py`; en *Advanced settings*, Python 3.12.
3. El historial SQLite se reinicia en cada redespliegue (limitación de la nube gratuita).
