"""Bucle de entrenamiento, predicción y métricas para clasificación binaria de abandono (PASOS 3 y 4)."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from torch import nn

from .config import SEED


def fijar_semillas(seed: int = SEED) -> None:
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


@torch.no_grad()
def predecir(modelo, loader) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (y, probabilidad de abandono)."""
    modelo.eval()
    ys, ps = [], []
    for seq, est, y in loader:
        ps.append(torch.sigmoid(modelo(seq, est))); ys.append(y)
    return torch.cat(ys).numpy().astype(int), torch.cat(ps).numpy()


def entrenar(modelo, dl_train, dl_val, epocas: int = 80, lr: float = 2e-3, weight_decay: float = 1e-3,
             pos_weight: float = 1.0, paciencia: int = 12, verbose: bool = True, nombre: str = ""):
    """AdamW + ReduceLROnPlateau + early stopping sobre el ROC-AUC de VALIDACIÓN.
    Pérdida de entrenamiento: BCE con pos_weight (desbalance); la de validación se reporta sin ponderar.
    Devuelve el historial por época y deja en el modelo los pesos de la mejor época."""
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    crit_val = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)
    hist, mejor, mejor_estado, sin_mejora = [], -1.0, copy.deepcopy(modelo.state_dict()), 0
    for ep in range(1, epocas + 1):
        t0 = time.time()
        modelo.train()
        perd, n, ys, ps = 0.0, 0, [], []
        for seq, est, y in dl_train:
            logit = modelo(seq, est)
            loss = crit(logit, y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
            perd += loss.item() * len(y); n += len(y)
            ys.append(y); ps.append(torch.sigmoid(logit.detach()))
        auc_tr = roc_auc_score(torch.cat(ys).numpy(), torch.cat(ps).numpy())
        yv, pv = predecir(modelo, dl_val)
        loss_v = crit_val(torch.logit(torch.tensor(pv).clamp(1e-6, 1 - 1e-6)), torch.tensor(yv, dtype=torch.float32)).item()
        auc_v = roc_auc_score(yv, pv)
        sched.step(auc_v)
        hist.append(dict(epoca=ep, loss_train=perd / n, loss_val=loss_v, auc_train=auc_tr, auc_val=auc_v,
                         acc_val=accuracy_score(yv, pv >= 0.5), lr=opt.param_groups[0]["lr"], seg=time.time() - t0))
        if verbose:
            print(f"[{nombre}] ép {ep:02d} | loss {perd / n:.3f}/{loss_v:.3f} | AUC {auc_tr:.3f}/{auc_v:.3f} "
                  f"| lr {opt.param_groups[0]['lr']:.1e} | {time.time() - t0:.1f}s")
        if auc_v > mejor + 1e-4:
            mejor, mejor_estado, sin_mejora = auc_v, copy.deepcopy(modelo.state_dict()), 0
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                if verbose:
                    print(f"  early stopping en la época {ep} (mejor AUC val = {mejor:.4f})")
                break
    modelo.load_state_dict(mejor_estado)
    return hist


def umbral_youden(y: np.ndarray, p: np.ndarray) -> float:
    """Misma regla que el ACA 1: mediana de los umbrales (rejilla 0,10–0,90) cuyo índice de Youden está a
    ≤ 0,005 del máximo. Se calcula en VALIDACIÓN, nunca en prueba."""
    umbrales = np.round(np.arange(0.10, 0.91, 0.01), 2)
    j = np.array([recall_score(y, p >= u, zero_division=0) + ((p < u) & (y == 0)).sum() / (y == 0).sum() - 1
                  for u in umbrales])
    return float(round(np.median(umbrales[np.isclose(j, j.max(), atol=0.005)]), 2))


def metricas(y: np.ndarray, p: np.ndarray, umbral: float) -> dict:
    pred = (p >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "especificidad": tn / (tn + fp), "f1": f1_score(y, pred),
            "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p),
            "VP": int(tp), "FP": int(fp), "FN": int(fn), "VN": int(tn)}


def bootstrap(y: np.ndarray, probs: dict, umbrales: dict, ref: str, n: int = 2000, seed: int = SEED) -> dict:
    """IC 95 % por bootstrap para cada modelo y diferencia PAREADA de ROC-AUC contra `ref`
    (mismos remuestreos de clientes para todos los modelos)."""
    rng = np.random.default_rng(seed)
    claves = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")
    res = {m: {k: [] for k in claves} for m in probs}
    delta = {m: [] for m in probs if m != ref}
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        aucs = {}
        for m, p in probs.items():
            mt = metricas(y[idx], p[idx], umbrales[m])
            for k in claves:
                res[m][k].append(mt[k])
            aucs[m] = mt["roc_auc"]
        for m in delta:
            delta[m].append(aucs[m] - aucs[ref])
    ic = {m: {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in r.items()} for m, r in res.items()}
    dif = {m: {"delta_auc": float(np.mean(v)), "ic95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
               "p_valor": float(min(1.0, 2 * min(np.mean(np.array(v) <= 0), np.mean(np.array(v) >= 0))))}
           for m, v in delta.items()}
    return {"ic95": ic, "vs_" + ref: dif}


def curva_roc(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return fpr, tpr
