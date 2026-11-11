"""Arma entregables/ENTREGABLES_ACA2_VENTAS/ según la SECCIÓN 4 «Entregables Oficiales del Proyecto»
del enunciado ACA_2_Aprendizaje_Automatico_CUN:

  1_Notebook/        notebook ejecutado (preprocesamiento, redes, entrenamiento, curvas y evaluación)
  2_Poster/          póster tipo infografía en PDF (A3, 1 página, enlaces clicables)
  3_App_Streamlit/   app.py, scripts auxiliares (utils/), modelos, base SQLite, requirements e instrucciones
  4_Video/           sólo si existe entregables/Video_Sustentacion_ACA2_Ventas.mp4

También deja ENTREGABLES_ACA2_VENTAS.zip al lado de la carpeta, listo para subir.

Uso:  python entregables/armar_entregables.py
Ejecutar después de generar_poster.py y generar_documentos.py para que las copias lleven las cifras finales.
"""
import base64
import re
import shutil
import sqlite3
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
ENT = BASE / "entregables"
DEST = ENT / "ENTREGABLES_ACA2_VENTAS"
VIDEO = ENT / "Video_Sustentacion_ACA2_Ventas.mp4"
IGNORAR = shutil.ignore_patterns("__pycache__", "*.pyc", "_rapido", "raw", "*.db-wal", "*.db-shm", "ventas.db")


def copiar(origen: Path, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    if origen.is_dir():
        shutil.copytree(origen, destino, ignore=IGNORAR, dirs_exist_ok=True)
    else:
        shutil.copy2(origen, destino)


# Se borran los archivos de la versión anterior; las carpetas vacías que Windows tenga bloqueadas
# (p. ej. abiertas en el Explorador) se dejan y se reutilizan.
DEST.mkdir(parents=True, exist_ok=True)
for p in sorted(DEST.rglob("*"), key=lambda x: len(x.parts), reverse=True):
    try:
        p.rmdir() if p.is_dir() else p.unlink()
    except OSError:
        if not p.is_dir():
            raise

# 1. Notebook de Jupyter
N1 = DEST / "1_Notebook"
copiar(BASE / "ACA2_Deep_Learning_Ventas.ipynb", N1 / "ACA2_Deep_Learning_Ventas.ipynb")
copiar(BASE / "requirements-notebook.txt", N1 / "requirements-notebook.txt")

# 2. Póster tipo infografía
copiar(ENT / "Poster_ACA2_Ventas_DL.pdf", DEST / "2_Poster" / "Poster_ACA2_Ventas_DL.pdf")
# Versión HTML autocontenida: las figuras van incrustadas en base64, así se ve bien desde cualquier carpeta
# (el poster.html original usa rutas ../figuras/ que sólo existen dentro del repositorio).
html = re.sub(r'src="\.\./figuras/([^"]+)"',
              lambda m: 'src="data:image/png;base64,' + base64.b64encode((BASE / "figuras" / m.group(1)).read_bytes()).decode() + '"',
              (ENT / "poster.html").read_text(encoding="utf-8"))
assert "../figuras/" not in html
(DEST / "2_Poster" / "Poster_ACA2_Ventas_DL.html").write_text(html, encoding="utf-8")

# 3. App interactiva Streamlit (carpeta autocontenida: se ejecuta tal cual)
N3 = DEST / "3_App_Streamlit"
for f in ("app.py", "requirements.txt", "README.md"):
    copiar(BASE / f, N3 / f)
for d in ("utils", "modelos", "aca1", "data", "figuras"):
    copiar(BASE / d, N3 / d)
# La base se copia con la API de respaldo de SQLite (incluye lo pendiente en el WAL) y con el historial vacío
src, dst = sqlite3.connect(BASE / "data" / "ventas.db"), sqlite3.connect(N3 / "data" / "ventas.db")
src.backup(dst)
src.close()
dst.execute("DELETE FROM predicciones")
dst.commit()
dst.execute("PRAGMA journal_mode=DELETE")   # un solo archivo .db, sin -wal/-shm
dst.execute("VACUUM")
dst.close()
(N3 / "COMO_EJECUTAR.txt").write_text(
    "App Streamlit · ACA 2 Ventas (abandono de clientes con GRU, LSTM y Transformer)\n\n"
    "Contenido\n"
    "  app.py            interfaz Streamlit\n"
    "  utils/            scripts auxiliares (preprocesamiento, modelos, inferencia, base de datos)\n"
    "  modelos/          pesos de las redes (.pt), escalador.json, Random Forest del ACA 1 y metricas.json\n"
    "  data/ventas.db    base SQLite (transacciones, clientes, series semanales, historial de predicciones)\n"
    "  aca1/             modelo y métricas originales del ACA 1 (referencia de comparación)\n"
    "  figuras/          gráficas que muestra la pestaña Desempeño\n\n"
    "Ejecutar (Python 3.12, desde esta carpeta)\n"
    "  python -m venv .venv\n"
    "  .venv\\Scripts\\python -m pip install -r requirements.txt\n"
    "  .venv\\Scripts\\streamlit run app.py\n\n"
    "Carga del modelo: app.py lee los pesos de modelos/*.pt con utils/modelos.cargar() y aplica el mismo\n"
    "preprocesamiento del notebook (utils/secuencias.py + modelos/escalador.json). No hay que entrenar nada.\n\n"
    f"App en línea: {(ENT / 'enlace_app.txt').read_text(encoding='utf-8').strip()}\n"
    f"Repositorio:  {(ENT / 'enlace_repo.txt').read_text(encoding='utf-8').strip()}\n", encoding="utf-8")

# 4. Video de sustentación (sólo si ya existe)
if VIDEO.exists():
    copiar(VIDEO, DEST / "4_Video" / VIDEO.name)

(DEST / "LEEME.txt").write_text(
    "ACA N° 2 – Aprendizaje Automático (CUN) · Pilar Ventas\n"
    "Abandono de clientes con secuencias semanales: GRU, LSTM y Transformer vs. Random Forest del ACA 1\n"
    "Integrantes: Juan Guillermo Angel Jaramillo · Luis Bernardo Correa Restrepo · María Alejandra Manga Betancourt\n"
    "Docente: Hamilton Rivera Flor\n\n"
    "Entregables oficiales (Sección 4 del enunciado)\n"
    "1_Notebook/       1. Notebook de Jupyter ejecutado: preprocesamiento, redes, entrenamiento, curvas y evaluación.\n"
    "                     Para re-ejecutarlo, clonar el repositorio (necesita utils/, aca1/ y data/).\n"
    "2_Poster/         2. Póster tipo infografía (PDF A3, 1 página, con enlaces a la app, el video y el repositorio)\n"
    "                     y Poster_ACA2_Ventas_DL.html, la misma versión en HTML con las figuras incrustadas.\n"
    "3_App_Streamlit/  3. App interactiva: app.py, scripts auxiliares, modelos, requirements.txt e instrucciones\n"
    "                     (COMO_EJECUTAR.txt y README.md).\n"
    + ("4_Video/          4. Video de sustentación (3 a 5 min).\n" if VIDEO.exists()
       else "                  4. Video de sustentación: se entrega por separado.\n"), encoding="utf-8")

zip_base = ENT / DEST.name
Path(f"{zip_base}.zip").unlink(missing_ok=True)
shutil.make_archive(str(zip_base), "zip", root_dir=ENT, base_dir=DEST.name)

for p in sorted(DEST.rglob("*")):
    if p.is_file() and not any(s in p.parts for s in ("utils", "modelos", "aca1", "figuras")):
        print(f"  {p.relative_to(DEST)}  ({p.stat().st_size / 1e6:.2f} MB)")
print(f"{zip_base}.zip  ({Path(f'{zip_base}.zip').stat().st_size / 1e6:.1f} MB)")
