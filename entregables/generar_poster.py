"""Genera el póster (A3 vertical) a partir de modelos/metricas.json y lo exporta a PDF con Edge/Chrome headless.

Uso:  python entregables/generar_poster.py [--app URL] [--video URL] [--repo URL]
Los enlaces pasados se guardan en entregables/enlace_*.txt y se reutilizan en las siguientes ejecuciones.
"""
import argparse
import json
import math
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "entregables"
PENDIENTE = {"app": "(enlace de Streamlit Cloud – por publicar)", "video": "(enlace del video – por publicar)",
             "repo": "(enlace del repositorio GitHub – por publicar)"}

ap = argparse.ArgumentParser()
for k in PENDIENTE:
    ap.add_argument(f"--{k}")
args = ap.parse_args()
for k, texto in PENDIENTE.items():
    archivo = OUT / f"enlace_{k}.txt"
    valor = getattr(args, k)
    if valor:
        archivo.write_text(valor + "\n", encoding="utf-8")
    elif archivo.exists() and archivo.read_text(encoding="utf-8").strip():
        valor = archivo.read_text(encoding="utf-8").strip()
    setattr(args, k, valor or texto)


def enlace(valor: str) -> str:
    """Hipervínculo clicable en el HTML y en el PDF; texto plano mientras no haya URL."""
    if valor.startswith(("http://", "https://")):
        return f'<a href="{valor}">{valor}</a>'
    return f"<i>{valor}</i>"


m = json.loads((BASE / "modelos" / "metricas.json").read_text(encoding="utf-8"))
R = m["resultados_test"]
APP = m["modelo_app"]
RF = "Random Forest (ACA 1)"
app, rf = R[APP], R[RF]
d_app = m["vs_rf"][APP]
n = m["n"]
pct = lambda x: f"{100 * x:.1f}".replace(".", ",") + " %"
num = lambda x, d=3: f"{x:.{d}f}".replace(".", ",")
miles = lambda x: f"{x:,}".replace(",", ".")
sgn = lambda x: ("−" if math.copysign(1, x) < 0 else "+") + num(abs(x))   # conserva el signo de −0,0000x
ic = app["ic95"]
ruta_fig = lambda f: f"../figuras/{f}"  # ruta relativa: funciona en cualquier equipo y en GitHub


def fila(k, v):
    d = m["vs_rf"].get(k)
    dif = f"{sgn(d['delta_auc'])} [{sgn(d['ic95'][0])}; {sgn(d['ic95'][1])}]" if d else "—"
    par = f"{v['parametros'] / 1e3:.1f} k".replace(".", ",") if v.get("parametros") else "—"
    return (f"<tr{' class=sel' if k == APP else ''}><td>{k}</td><td>{par}</td>"
            f"<td>{num(m['seleccion_val'][k]['auc_val']) if k in m['seleccion_val'] else '—'}</td>"
            f"<td>{num(v['roc_auc'])}</td><td>{num(v['f1'])}</td><td>{num(v['recall'])}</td><td>{num(v['precision'])}</td><td>{dif}</td></tr>")


filas = "".join(fila(k, v) for k, v in R.items())


def figura_poster():
    """Curvas del modelo de la app (todas las semillas) + matriz de confusión, con letra grande para el póster."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    hs = m["historial"][APP]
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.9), gridspec_kw={"width_ratios": [1, 1, .75]})
    for i, h in enumerate(hs):
        ep = [x["epoca"] for x in h]
        lw, al = (2.2, 1) if i == 0 else (1, .35)
        ax[0].plot(ep, [x["loss_train"] for x in h], color="#2E86AB", lw=lw, alpha=al, label="entrenamiento" if i == 0 else None)
        ax[0].plot(ep, [x["loss_val"] for x in h], color="#E07A1F", lw=lw, alpha=al, label="validación" if i == 0 else None)
        ax[1].plot(ep, [x["auc_train"] for x in h], color="#2E86AB", lw=lw, alpha=al, label="entrenamiento" if i == 0 else None)
        ax[1].plot(ep, [x["auc_val"] for x in h], color="#E07A1F", lw=lw, alpha=al, label="validación" if i == 0 else None)
    ax[1].axhline(rf["roc_auc"], color="gray", ls=":", label="RF ACA 1 (prueba)")
    ax[0].set(title="Pérdida BCE por época", xlabel="Época"); ax[1].set(title="ROC-AUC por época", xlabel="Época")
    ax[0].legend(fontsize=9); ax[1].legend(fontsize=9)
    c = np.array([[app["VN"], app["FP"]], [app["FN"], app["VP"]]])
    ax[2].imshow(c, cmap="Oranges")
    for (i, j), v in np.ndenumerate(c):
        ax[2].text(j, i, v, ha="center", va="center", fontsize=16, color="white" if v > c.max() / 2 else "black")
    et = ["Recompra", "Abandono"]
    ax[2].set_xticks(range(2), et); ax[2].set_yticks(range(2), et)
    ax[2].set(title=f"Matriz de confusión (prueba, umbral {app['umbral']:.2f})", xlabel="Predicho", ylabel="Real")
    ax[2].grid(False)
    ruta = BASE / "figuras" / "13_poster_modelo_app.png"
    fig.savefig(ruta, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return ruta


figura_poster()
preview = BASE / "figuras" / "12_app_preview.png"

ARQ_TXT = {
    "GRU": ("GRU (red recurrente con compuertas) + R-F-M", "11_arquitectura_gru.png",
            "Lee las 40 semanas en orden; sus compuertas de actualización y reinicio deciden qué recordar del pasado. El último "
            "estado oculto (32) se concatena con R-F-M y pasa a una capa densa con salida sigmoide."),
    "LSTM": ("LSTM (memoria a largo y corto plazo) + R-F-M", "11_arquitectura_lstm.png",
             "Lee las 40 semanas en orden con una celda de memoria y tres compuertas (entrada, olvido, salida). El último estado "
             "oculto (32) se concatena con R-F-M y pasa a una capa densa con salida sigmoide."),
    "Transformer": ("Encoder Transformer temporal + R-F-M", "11_arquitectura_transformer.png",
                    "Cada semana es un «token». La autoatención (4 cabezas, 2 bloques) relaciona todas las semanas entre sí y el "
                    "token [CLS] resume al cliente; su atención muestra qué semanas mira la red."),
}
t_arq, img_arq, nota_arq = ARQ_TXT[APP]
nota_arq += (f" {m['resultados_test'][APP]['parametros']:,} parámetros por red · ensamble de {len(m['semillas'])} semillas · "
             "AdamW, early stopping por AUC de validación.").replace(",", ".")
ARQ_HTML = (f'<div class="card mb"><h2>Arquitectura elegida para la app (mayor ROC-AUC en validación): {t_arq}</h2>'
            f'<img src="{ruta_fig(img_arq)}" style="max-height:46mm"><p class=note style="margin:0">{nota_arq}</p></div>')

CONCLUSIONES = json.loads((OUT / "conclusiones.json").read_text(encoding="utf-8"))
imp = m["importancia"]

html = f"""<!doctype html><html lang=es><head><meta charset=utf-8><title>Póster ACA 2 – Abandono de clientes con Deep Learning</title><style>
@page{{size:297mm 420mm;margin:0}}
*{{box-sizing:border-box}}html{{background:#cfd6e0}}body{{margin:6mm auto;width:297mm;min-height:420mm;font-family:'Segoe UI','DejaVu Sans',sans-serif;color:#1d2433;background:#f6f4f1;font-size:9.1pt;box-shadow:0 2px 12px #0004}}@media print{{html{{background:none}}body{{margin:0;height:420mm;overflow:hidden;box-shadow:none}}}}
header{{background:linear-gradient(135deg,#1d2f4f,#1B7F79);color:#fff;padding:6mm 10mm 4.5mm;display:flex;gap:6mm;align-items:center}}header .tit{{flex:1}}header .lk a{{color:#fff;text-decoration:underline}}header .lk{{width:80mm;background:#E07A1F;border-radius:2.5mm;padding:2.5mm 3.5mm;font-size:8pt;line-height:1.35;word-break:break-all}}
h1{{margin:0;font-size:19pt;line-height:1.12}}header p{{margin:2mm 0 0;font-size:9pt;opacity:.93}}
.wrap{{padding:4mm 10mm}}.row{{display:flex;gap:4mm;margin-bottom:3mm}}.mb{{margin-bottom:3mm}}
.card{{background:#fff;border-radius:3mm;padding:2.8mm 3.8mm;box-shadow:0 1px 3px #0002;flex:1;min-width:0}}
.card h2{{margin:0 0 2mm;font-size:11.5pt;color:#1d2f4f;border-left:2mm solid #E07A1F;padding-left:2.5mm}}
.card p,.card li{{margin:0 0 1.2mm;line-height:1.33}}ul{{margin:0;padding-left:4.5mm}}
.steps{{display:flex;gap:2.5mm}}.step{{flex:1;background:#f7f9fc;border-top:2mm solid #1B7F79;border-radius:2mm;padding:2.2mm;font-size:8.3pt;line-height:1.3}}
.step>b:first-child{{display:block;color:#E07A1F;font-size:9pt;margin-bottom:.8mm}}
.kpis{{display:flex;gap:2.5mm;margin-bottom:2.5mm}}.kpi{{flex:1;background:#1d2f4f;color:#fff;border-radius:2mm;padding:2mm;text-align:center}}
.kpi span{{display:block;font-size:16pt;font-weight:bold;color:#ffd166}}.kpi small{{font-size:7.6pt}}
img{{max-width:100%;display:block;margin:0 auto}}
table{{border-collapse:collapse;width:100%;font-size:8.2pt}}th,td{{padding:1.1mm 1.3mm;border-bottom:1px solid #dde3ec;text-align:center}}
th{{background:#eef2f8;color:#1d2f4f}}td:first-child,th:first-child{{text-align:left}}tr.sel td{{font-weight:bold;background:#fff4d6}}
.ref{{font-size:6.9pt;line-height:1.22;padding-left:5mm;text-indent:-5mm;margin:0 0 1mm}}
.note{{font-size:7.8pt;color:#555}}.ref a{{color:#1d4f91;overflow-wrap:break-word}}
.links{{background:#1d2f4f;color:#fff}}.links h2{{color:#fff;border-color:#E07A1F}}.links p{{font-size:9pt;line-height:1.5;margin:0}}.links a{{color:#ffd166;text-decoration:underline;word-break:break-all}}
</style></head><body>
<header><div class=tit><h1>Del ML clásico al Deep Learning: predicción del abandono de clientes a partir de su secuencia semanal de compras con GRU, LSTM y Transformer</h1>
<p>ACA N° 2 · Aprendizaje Automático · Especialización en Inteligencia Artificial · CUN – Escuela de Ingeniería · Docente: Hamilton Rivera Flor<br>
<b>Integrantes:</b> Juan Guillermo Angel Jaramillo · Luis Bernardo Correa Restrepo · María Alejandra Manga Betancourt</p></div>
<div class=lk><b>App:</b> {enlace(args.app)}<br><b>Video:</b> {enlace(args.video)}<br><b>Código:</b> {enlace(args.repo)}</div></header>
<div class=wrap>
<div class=row>
<div class=card><h2>Objetivo general</h2><p>Entrenar redes neuronales secuenciales (GRU, LSTM y Transformer temporal) que lean la <b>historia semanal de compras</b> de cada cliente para anticipar su abandono, compararlas con el Random Forest del ACA 1 en los mismos clientes y desplegar la mejor en una app Streamlit explicable.</p></div>
<div class=card><h2>Situación problema</h2><p><i>¿Cómo aprovechar redes neuronales (CNN, RNN, Transformers) para extraer patrones complejos en datos de ventas y llevarlos a una interfaz interactiva?</i></p><p>En el ACA 1 el modelo veía una <b>foto</b> del cliente (recencia, frecuencia, gasto). Ahora la red ve la <b>película</b>: 40 semanas de compras, con su orden y su ritmo.</p></div>
<div class=card><h2>Dataset: UCI Online Retail (Chen, 2015)</h2><ul>
<li>{miles(n['lineas_originales'])} líneas de factura (dic-2010 → dic-2011) de un minorista en línea del Reino Unido; {miles(n['lineas_limpias'])} ventas limpias.</li>
<li>{miles(n['clientes'])} clientes con compras antes del corte del 1-sep-2011; <b>abandono</b> = no vuelve a comprar después ({pct(n['tasa_abandono'])}).</li>
<li>Partición idéntica al ACA 1: train {n['train']} · val {n['val']} · prueba {n['test']}.</li></ul></div>
</div>
<div class="card mb"><h2>Ruta metodológica</h2><div class=steps>
<div class=step><b>1 · Exploración y SQL</b>Transacciones en <b>SQLite</b>; calidad, EDA y limpieza con consultas y <b>vistas SQL</b> (nulos, duplicados, cancelaciones, precios ≤ 0).</div>
<div class=step><b>2 · Pipeline</b><b>Secuencias</b> de 40 semanas × 7 canales; escalado sólo con train; partición por cliente sin fuga; <b>aumento</b> (escalado, jitter, recorte de ventana); pos_weight.</div>
<div class=step><b>3 · Modelado</b>RF del ACA 1 (base) · <b>GRU</b> · <b>LSTM</b> · <b>Transformer</b> temporal; 5 semillas; AdamW y early stopping por AUC de validación.</div>
<div class=step><b>4 · Evaluación</b>Curvas por época, ROC/PR, matriz de confusión, IC 95 % bootstrap, ΔAUC pareado, permutación, oclusión y atención.</div>
<div class=step><b>5 · Despliegue</b>App Streamlit: cliente, secuencia manual, lotes CSV, historial y consultas SQL, desempeño y selector de modelos.</div></div></div>
{ARQ_HTML}
<div class=row>
<div class=card style="flex:1.25"><h2>Resultados en prueba (n = {n['test']}, los mismos clientes del ACA 1)</h2>
<div class=kpis><div class=kpi><span>{num(app['roc_auc'])}</span><small>ROC-AUC {APP}</small></div><div class=kpi><span>{num(rf['roc_auc'])}</span><small>ROC-AUC RF ACA 1</small></div>
<div class=kpi><span>{pct(app['recall'])}</span><small>Recall abandono</small></div><div class=kpi><span>{num(app['f1'])}</span><small>F1</small></div></div>
<table><tr><th>Modelo</th><th>Parám.</th><th>AUC val</th><th>ROC-AUC</th><th>F1</th><th>Recall</th><th>Precisión</th><th>ΔAUC vs RF [IC 95 %]</th></tr>{filas}</table>
<p class=note style="margin-top:1.5mm">IC 95 % (bootstrap, 2.000 remuestreos pareados) de {APP}: ROC-AUC {num(ic['roc_auc'][0])}–{num(ic['roc_auc'][1])} · F1 {num(ic['f1'][0])}–{num(ic['f1'][1])} · recall {num(ic['recall'][0])}–{num(ic['recall'][1])}. ΔAUC frente al RF: {sgn(d_app['delta_auc'])} (p = {num(d_app['p_valor'])}). Umbrales elegidos en validación (regla de Youden del ACA 1). <b>Al mismo punto de operación</b> (recall ≥ 0,80 fijado en validación): {APP} {m['politica_recall80'][APP]['VP']} abandonos detectados con {m['politica_recall80'][APP]['FP']} falsas alarmas; RF {m['politica_recall80'][RF]['VP']} con {m['politica_recall80'][RF]['FP']}.</p></div>
<div class=card><h2>Vista previa de la app Streamlit</h2>{f'<img src="../figuras/{preview.name}" style="max-height:60mm">' if preview.exists() else '<p>(captura pendiente)</p>'}</div>
</div>
<div class="card mb"><h2>{APP}: curvas de entrenamiento/validación (5 semillas) y matriz de confusión</h2>
<img src="{ruta_fig('13_poster_modelo_app.png')}" style="max-height:50mm"></div>
<div class=row>
<div class=card style="flex:1.35"><h2>Interpretabilidad</h2><img src="{ruta_fig('10_interpretabilidad.png')}" style="max-height:46mm">
<p class=note>Lo más influyente para {APP} es <b>{imp[0]['variable']}</b> (caída de AUC {num(imp[0]['caida_auc'])}), luego {imp[1]['variable']} y {imp[2]['variable']}. Borrar una compra de las 8 últimas semanas cambia P(abandono) en {sgn(m['oclusion_ultimas8'])} frente a {sgn(m['oclusion_primer_trimestre'])} en el primer trimestre. El [CLS] del Transformer dedica {pct(m['atencion_ultimas8'])} de su atención a esas 8 semanas (20 % si fuera uniforme).</p></div>
<div class=card style="flex:1"><h2>Conclusiones</h2><ul>{''.join(f'<li>{c}</li>' for c in CONCLUSIONES[:6])}</ul></div>
</div>
<div class="card mb" style="columns:2;column-gap:6mm"><h2 style="column-span:all">Referencias (APA 7)</h2>
<p class=ref>Chen, D. (2015). <i>Online Retail</i> [Conjunto de datos]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33</p>
<p class=ref>Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., &amp; Bengio, Y. (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. En <i>Proceedings of EMNLP 2014</i> (pp. 1724–1734). https://doi.org/10.3115/v1/D14-1179</p>
<p class=ref>Hochreiter, S., &amp; Schmidhuber, J. (1997). Long short-term memory. <i>Neural Computation, 9</i>(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735</p>
<p class=ref>Iwana, B. K., &amp; Uchida, S. (2021). An empirical survey of data augmentation for time series classification with neural networks. <i>PLOS ONE, 16</i>(7), e0254841. https://doi.org/10.1371/journal.pone.0254841</p>
<p class=ref>Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., &amp; Polosukhin, I. (2017). Attention is all you need. En <i>Advances in Neural Information Processing Systems 30</i> (pp. 5998–6008). https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html</p>
</div>
<div class="card links"><h2>Enlaces del proyecto</h2><p><b>App Streamlit:</b> {enlace(args.app)}<br><b>Video de sustentación:</b> {enlace(args.video)}<br><b>Código (GitHub):</b> {enlace(args.repo)}</p></div>
</div></body></html>"""
# Todas las URL de las referencias como hipervínculos clicables en el PDF
html = re.sub(r'(?<=[ >])(https://(?:doi\.org|proceedings\.neurips\.cc)/[^\s<]+)', r'<a href="\1">\1</a>', html)

ruta_html = OUT / "poster.html"
navegador = next((p for p in (shutil.which("msedge"), shutil.which("chrome"),
                              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                              r"C:\Program Files\Google\Chrome\Application\chrome.exe") if p and Path(p).exists()), None)


def escribir(zoom: float) -> None:
    ruta_html.write_text(html.replace("<style>", f"<style>.wrap{{zoom:{zoom:.2f}}}", 1), encoding="utf-8")


def paginas(pdf: Path) -> int:
    import pypdfium2
    doc = pypdfium2.PdfDocument(str(pdf))
    k = len(doc)
    doc.close()
    return k


escribir(1.0)
if navegador:
    pdf = OUT / "Poster_ACA2_Ventas_DL.pdf"
    zoom = 1.0
    while True:  # reduce la escala hasta que el póster quepa en una sola hoja A3
        escribir(zoom)
        perfil = tempfile.mkdtemp(prefix="poster_")   # perfil propio: no interfiere con un navegador abierto
        pdf.unlink(missing_ok=True)                    # evita dar por buena una versión anterior
        r = subprocess.run([navegador, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={perfil}",
                            "--allow-file-access-from-files", "--virtual-time-budget=10000", f"--print-to-pdf={pdf}",
                            ruta_html.resolve().as_uri()], capture_output=True, text=True, timeout=180)
        for _ in range(20):                           # el navegador puede terminar de escribir tras salir
            if pdf.exists() and pdf.stat().st_size > 0:
                break
            time.sleep(0.5)
        if not pdf.exists():
            raise RuntimeError(f"El navegador no generó el PDF:\n{r.stderr[-1500:]}")
        if paginas(pdf) <= 1 or zoom <= 0.8:
            break
        zoom -= 0.02
    print(f"HTML: {ruta_html}\nPDF: {pdf} (escala {zoom:.2f}, {paginas(pdf)} página/s)")
else:
    print("No se encontró Edge/Chrome: abre poster.html e imprímelo a PDF (A3 vertical, sin márgenes).")
