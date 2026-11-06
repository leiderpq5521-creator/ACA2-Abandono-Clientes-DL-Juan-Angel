"""Reúne en entregables/ENTREGABLES_ACA2_VENTAS/ los archivos que se entregan (misma estructura que el ACA 1).

Uso:  python entregables/armar_entregables.py
Ejecutar después de generar_poster.py y generar_documentos.py para que las copias lleven las cifras finales.
"""
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DEST = BASE / "entregables" / "ENTREGABLES_ACA2_VENTAS"
ARCHIVOS = [
    BASE / "ACA2_Deep_Learning_Ventas.ipynb",                 # 1. Notebook ejecutado
    BASE / "entregables" / "Poster_ACA2_Ventas_DL.pdf",      # 2. Póster tipo infografía (A3, 1 página)
    BASE / "entregables" / "poster.html",                    #    fuente HTML del póster
    BASE / "app.py",                                         # 3. App Streamlit
    BASE / "requirements.txt",                               #    dependencias de la app
    BASE / "README.md",                                      #    instrucciones y resultados
    BASE / "entregables" / "Guion_Video_3-5min.md",          # 4. Guion del video de sustentación
]
VIDEO = "Video_Sustentacion_ACA2_Ventas.mp4"                 # se copia si existe en entregables/

DEST.mkdir(parents=True, exist_ok=True)
copiados = []
for f in ARCHIVOS + [BASE / "entregables" / VIDEO]:
    if f.exists():
        shutil.copy2(f, DEST / f.name)
        copiados.append(f.name)
# poster.html usa rutas ../figuras/: en la copia se ajustan para que siga mostrando las imágenes
html = DEST / "poster.html"
if html.exists():
    html.write_text(html.read_text(encoding="utf-8").replace('src="../figuras/', 'src="../../figuras/'), encoding="utf-8")
(DEST / "LEEME.txt").write_text(
    "ACA N° 2 – Aprendizaje Automático (CUN) · Pilar Ventas\n"
    "Abandono de clientes con secuencias semanales: GRU, LSTM y Transformer vs. Random Forest del ACA 1\n"
    "Integrantes: Juan Guillermo Angel Jaramillo · Luis Bernardo Correa Restrepo · María Alejandra Manga Betancourt\n"
    "Docente: Hamilton Rivera Flor\n\n"
    "Contenido:\n"
    "1. ACA2_Deep_Learning_Ventas.ipynb – notebook ejecutado (SQL, EDA, pipeline, redes, evaluación)\n"
    "2. Poster_ACA2_Ventas_DL.pdf – póster A3 con enlaces clicables (poster.html es su fuente)\n"
    "3. app.py + requirements.txt – app Streamlit (requiere utils/, modelos/ y data/ del repositorio)\n"
    "4. Guion_Video_3-5min.md – guion del video" + (f"; {VIDEO} – video de sustentación\n" if VIDEO in copiados else
                                                   " (video: ver enlace en el póster)\n") +
    "README.md – resultados, instrucciones de ejecución y despliegue\n", encoding="utf-8")
print(f"{DEST}\n" + "\n".join(f"  · {c}" for c in copiados + ["LEEME.txt"]))
