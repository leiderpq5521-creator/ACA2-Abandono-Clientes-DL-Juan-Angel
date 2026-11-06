"""Diagramas de las arquitecturas (figuras para notebook, póster y video)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .config import DIR_FIGURAS, N_SEMANAS

AZUL, NARANJA, GRIS, OSCURO = "#2E86AB", "#E07A1F", "#DDE6F0", "#1d2f4f"


def _lienzo(titulo):
    fig, ax = plt.subplots(figsize=(17, 3.6))
    ax.set_xlim(0, 17); ax.set_ylim(-1.15, 2.75); ax.axis("off")
    ax.set_title(titulo, fontsize=11.5, color=OSCURO, loc="left")
    return fig, ax


def _caja(ax, x, w, txt, col, sub="", h=1.9, y=1.0, fs=9.5):
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0.03,rounding_size=0.08", fc=col, ec=OSCURO, lw=1.2))
    ax.text(x + w / 2, y, txt, ha="center", va="center", fontsize=fs, fontweight="bold",
            color="white" if col in (AZUL, NARANJA) else OSCURO)
    if sub:
        ax.text(x + w / 2, y - h / 2 - .08, sub, ha="center", va="top", fontsize=8.5, color="#333")


def _flecha(ax, x0, x1, y=1.0, y1=None, col=OSCURO):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y if y1 is None else y1), arrowstyle="-|>", mutation_scale=12, color=col))


def _entrada(ax):
    # mini mapa de calor semanas × canales
    for t in range(12):
        for c in range(7):
            v = 0.15 + 0.7 * ((t * 7 + c * 3) % 5 == 0 or (t > 8 and c < 3))
            ax.add_patch(plt.Rectangle((.15 + t * .1, .35 + c * .19), .09, .17, fc=plt.cm.Oranges(v), ec="white", lw=.3))
    ax.text(.75, -.05, f"{N_SEMANAS} semanas × 7 canales\n(gasto, pedidos, unidades,\nproductos, devoluciones,\ndías sin comprar, es cliente)",
            ha="center", va="top", fontsize=8, color="#333")


def _salida(ax, x0):
    _caja(ax, x0, 1.25, "Concatenar\n+ R-F-M\n(3 var.)", GRIS, "derivados de la\nmisma secuencia", h=1.4)
    _flecha(ax, x0 + 1.3, x0 + 1.5)
    _caja(ax, x0 + 1.55, 1.25, "Densa 32\nGELU +\nDropout", NARANJA, h=1.4)
    _flecha(ax, x0 + 2.85, x0 + 3.05)
    _caja(ax, x0 + 3.1, 1.45, "Sigmoide\nP(abandono)", NARANJA, "umbral Youden\n(validación)", h=1.4, fs=8.8)


def dibujar_recurrente(tipo="GRU", ruta=None):
    ruta = ruta or DIR_FIGURAS / f"11_arquitectura_{tipo.lower()}.png"
    fig, ax = _lienzo(f"{tipo} de 2 capas sobre la secuencia semanal del cliente + R-F-M · ensamble de 5 semillas · "
                      "la celda recurrente decide qué recordar y qué olvidar semana a semana")
    _entrada(ax)
    _flecha(ax, 1.45, 1.75)
    for k, x in enumerate([1.8, 3.15, 4.5, 6.6]):
        lab = ["t = 1", "t = 2", "t = 3", f"t = {N_SEMANAS}"][k]
        _caja(ax, x, 1.0, f"{tipo}\ncapa 1", AZUL, h=.75, y=.55, fs=8.5)
        _caja(ax, x, 1.0, f"{tipo}\ncapa 2", AZUL, h=.75, y=1.55, fs=8.5)
        ax.text(x + .5, -.15, lab, ha="center", fontsize=8.5)
        _flecha(ax, x + .5, x + .5, y=.93, y1=1.17)
        if k < 3:
            nx = [3.15, 4.5, 6.6][k]
            for yy in (.55, 1.55):
                _flecha(ax, x + 1.02, nx - .02 if k < 2 else 5.55, y=yy)
    ax.text(6.05, 1.05, "…", fontsize=20, ha="center", color=OSCURO)
    for yy in (.55, 1.55):
        _flecha(ax, 5.95 if yy else 5.95, 6.58, y=yy)
    ax.text(4.25, 2.4, "estado oculto de 32 unidades que «lee» la película del cliente", ha="center", fontsize=9.5,
            color=AZUL, fontweight="bold")
    _flecha(ax, 7.65, 8.15, y=1.55, y1=1.0)
    _caja(ax, 8.2, 1.2, "Último\nestado\noculto", GRIS, "32", h=1.4)
    _flecha(ax, 9.45, 9.65)
    _salida(ax, 9.7)
    fig.savefig(ruta, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return ruta


def dibujar_transformer(ruta=DIR_FIGURAS / "11_arquitectura_transformer.png"):
    fig, ax = _lienzo("Encoder Transformer temporal · cada semana es un «token» · la autoatención relaciona todas las "
                      "semanas entre sí y el token [CLS] resume al cliente · ensamble de 5 semillas")
    _entrada(ax)
    _flecha(ax, 1.45, 1.75)
    _caja(ax, 1.8, 1.45, "Proyección\nlineal 7→32\n+ [CLS]\n+ posición", AZUL, f"{N_SEMANAS + 1} tokens × 32")
    _flecha(ax, 3.3, 3.5)
    ax.add_patch(FancyBboxPatch((3.55, -.12), 6.3, 2.25, boxstyle="round,pad=0.05,rounding_size=0.1", fc="#EEF4FA",
                                ec=AZUL, lw=1.5, ls="--"))
    x = 3.7
    for t, w in [("Layer\nNorm", 1.0), ("Atención\nmulti-cabeza\n(4 cabezas)", 1.4), ("+ residual", 1.0),
                 ("Layer\nNorm", 1.0), ("MLP\n32→64→32", 1.15)]:
        _caja(ax, x, w, t, AZUL, h=1.5, fs=8.3)
        x += w + .12
    ax.text(6.7, 2.3, "Bloque Transformer × 2 (pre-norm)", ha="center", fontsize=10.5, fontweight="bold", color=AZUL)
    _flecha(ax, 9.9, 10.1)
    _caja(ax, 10.15, 1.0, "Token\n[CLS]\n+ LN", GRIS, "32", h=1.4)
    _flecha(ax, 11.2, 11.4)
    _salida(ax, 11.45)
    xa = 3.7 + 1.0 + .12 + 1.4 / 2
    _flecha(ax, xa, xa, y=1.78, y1=2.08, col=NARANJA)
    ax.text(xa + .1, -.75, "Interpretabilidad: atención del [CLS] sobre cada semana y oclusión semana a semana",
            fontsize=9, color=NARANJA, fontweight="bold")
    fig.savefig(ruta, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return ruta


ARCHIVO = {"GRU": "11_arquitectura_gru.png", "LSTM": "11_arquitectura_lstm.png", "Transformer": "11_arquitectura_transformer.png"}


def dibujar_todas():
    return [dibujar_recurrente("GRU"), dibujar_recurrente("LSTM"), dibujar_transformer()]


if __name__ == "__main__":
    print(dibujar_todas())
