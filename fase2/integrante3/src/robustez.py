"""Robustez e interpretación (integrante 3). No calcula nada sobre test.

1. Ablation (H1, H2): mismo HistGradientBoosting con cuatro conjuntos de variables.
2. Rendimiento por trimestre y por volatilidad reciente (H3).
3. Análisis de errores: margen respecto al umbral y transiciones de clase.
"""
import json
import platform
import time
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from threadpoolctl import threadpool_limits

from comun import ROOT, SEED, GAP, experiment, cargar_desarrollo

OUT = ROOT / 'results'
FIG = OUT / 'figures'

# --- Grupos de variables (las 33 de PROTOCOLO.md, sin crear nuevas) ---------
TASA_ACTUAL = ['median_fee_rate', 'umbral_local', 'fee_relativa']
# Como en la H2 de P1, `total_fees` se considera actividad de la red, no comisión.
COMISIONES = TASA_ACTUAL + [
    'avg_fee_rate', 'fee_range_min', 'fee_range_max', 'ratio_avg_mediana',
    'fee_lag_1', 'fee_lag_6', 'fee_lag_144',
    'fee_media_6', 'fee_std_6', 'fee_media_144', 'fee_std_144', 'fee_media_1008', 'fee_std_1008']
ACTIVIDAD = [
    'total_fees', 'size', 'tx_count', 'difficulty', 'input_count', 'output_count', 'output_amount',
    'size_por_tx', 'input_count_por_tx', 'output_count_por_tx', 'size_relativo_1008',
    'intervalo_segundos', 'intervalo_negativo']
CALENDARIO = ['hora_sin', 'hora_cos', 'dia_semana_sin', 'dia_semana_cos']

CONJUNTOS = {
    'A0_tasa_actual': TASA_ACTUAL,
    'A_comisiones': COMISIONES,
    'B_comisiones_actividad': COMISIONES + ACTIVIDAD,
    'C_todas': COMISIONES + ACTIVIDAD + CALENDARIO,
}
ETIQUETAS = {
    'A0_tasa_actual': 'A0: tasa actual (3)',
    'A_comisiones': 'A: comisiones (16)',
    'B_comisiones_actividad': 'B: + actividad (29)',
    'C_todas': 'C: + calendario (33)',
    'persistencia': 'Persistencia',
}
COLORES = {'A0_tasa_actual': '#2a78d6', 'A_comisiones': '#eb6834',
           'B_comisiones_actividad': '#1baf7a', 'C_todas': '#eda100', 'persistencia': '#8a8984'}

# Configuración ganadora de integrante 2 (results/seleccion.json), fija en todos
# los conjuntos para aislar el efecto de las variables.
HGB = {'max_iter': 180, 'max_leaf_nodes': 15, 'learning_rate': 0.1,
       'min_samples_leaf': 80, 'l2_regularization': 5.0}
UMBRALES = np.round(np.arange(.10, .901, .025), 3)
N_BOOT, BLOQUE = 1000, 1008


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def elegir_umbral(y, p):
    """Mismo criterio que integrante 2: máximo F1 OOF; empate, el más cercano a 0.5."""
    s = [f1_score(y, p >= t, zero_division=0) for t in UMBRALES]
    i = max(range(len(UMBRALES)), key=lambda k: (s[k], -abs(UMBRALES[k] - .5)))
    return float(UMBRALES[i])


def metricas_binarias(y, pred):
    y, pred = np.asarray(y), np.asarray(pred).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {'f1': f1, 'precision': prec, 'recall': rec}


def metricas_segmento(y, p, pred):
    y = np.asarray(y)
    m = {'n': len(y), 'prevalencia': float(y.mean()), **metricas_binarias(y, pred)}
    m['average_precision'] = average_precision_score(y, p) if 0 < y.sum() < len(y) else np.nan
    return m


def ablation(X, y, a, b):
    folds = list(TimeSeriesSplit(n_splits=3, test_size=40000, gap=GAP).split(X.iloc[:a]))
    persist = y.shift(1).to_numpy()  # clase del bloque actual, conocida al predecir
    filas_fold, resumen, preds_val = [], [], {}
    for nombre, cols in CONJUNTOS.items():
        oof_y, oof_p = [], []
        for k, (tr, va) in enumerate(folds, 1):
            modelo = experiment.estimator('HistGradientBoosting', HGB)
            with threadpool_limits(limits=4):
                modelo.fit(X.iloc[tr][cols], y.iloc[tr])
                p = modelo.predict_proba(X.iloc[va][cols])[:, 1]
            yy = y.iloc[va].to_numpy()
            filas_fold.append({'conjunto': nombre, 'fold': k, 'inicio': int(va[0]), 'fin': int(va[-1]),
                               **metricas_segmento(yy, p, p >= .5)})
            oof_y.append(yy); oof_p.append(p)
            log(f'{nombre} fold {k}: AP={filas_fold[-1]["average_precision"]:.4f}')
        oof_y, oof_p = np.concatenate(oof_y), np.concatenate(oof_p)
        umbral = elegir_umbral(oof_y, oof_p)
        modelo = experiment.estimator('HistGradientBoosting', HGB)
        with threadpool_limits(limits=4):
            modelo.fit(X.iloc[:a - GAP][cols], y.iloc[:a - GAP])
            p_val = modelo.predict_proba(X.iloc[a:b][cols])[:, 1]
        preds_val[nombre] = (p_val, umbral)
        cv = pd.DataFrame([r for r in filas_fold if r['conjunto'] == nombre])
        resumen.append({'conjunto': nombre, 'n_variables': len(cols), 'umbral': umbral,
                        'cv_ap_media': cv.average_precision.mean(), 'cv_ap_std': cv.average_precision.std(),
                        'cv_f1_media_05': cv.f1.mean(), 'cv_f1_std_05': cv.f1.std(),
                        **{f'val_{k}': v for k, v in metricas_segmento(
                            y.iloc[a:b], p_val, p_val >= umbral).items() if k not in ('n', 'prevalencia')},
                        'val_roc_auc': roc_auc_score(y.iloc[a:b], p_val)})
        log(f'{nombre} validación: F1={resumen[-1]["val_f1"]:.4f} AP={resumen[-1]["val_average_precision"]:.4f}')
    # Referencia de persistencia en los mismos folds y en validación (solo como comparación).
    for k, (tr, va) in enumerate(folds, 1):
        filas_fold.append({'conjunto': 'persistencia', 'fold': k, 'inicio': int(va[0]), 'fin': int(va[-1]),
                           **metricas_segmento(y.iloc[va], persist[va], persist[va])})
    pv = persist[a:b]
    resumen.append({'conjunto': 'persistencia', 'n_variables': 0, 'umbral': np.nan,
                    **{f'val_{k}': v for k, v in metricas_segmento(y.iloc[a:b], pv, pv).items()
                       if k not in ('n', 'prevalencia')}})
    preds_val['persistencia'] = (pv.astype(float), .5)
    return pd.DataFrame(filas_fold), pd.DataFrame(resumen), preds_val


def bootstrap_diferencias(y, preds, pares):
    """Bootstrap por bloques móviles (1008 bloques) sobre validación, emparejado."""
    rng = np.random.default_rng(SEED)
    y = np.asarray(y); n = len(y); nb = int(np.ceil(n / BLOQUE))
    idx_all = [np.concatenate([np.arange(s, s + BLOQUE) for s in rng.integers(0, n - BLOQUE, nb)])[:n]
               for _ in range(N_BOOT)]
    filas = []
    for nuevo, base in pares:
        (p1, u1), (p0, u0) = preds[nuevo], preds[base]
        d_ap, d_f1 = [], []
        for idx in idx_all:
            yy = y[idx]
            d_ap.append(average_precision_score(yy, p1[idx]) - average_precision_score(yy, p0[idx]))
            d_f1.append(metricas_binarias(yy, p1[idx] >= u1)['f1'] - metricas_binarias(yy, p0[idx] >= u0)['f1'])
        for met, d, obs in [('average_precision', d_ap, average_precision_score(y, p1) - average_precision_score(y, p0)),
                            ('f1', d_f1, f1_score(y, p1 >= u1) - f1_score(y, p0 >= u0))]:
            filas.append({'comparacion': f'{nuevo} - {base}', 'metrica': met, 'diferencia': obs,
                          'ic95_inf': np.percentile(d, 2.5), 'ic95_sup': np.percentile(d, 97.5),
                          'prop_boot_mayor_0': float(np.mean(np.array(d) > 0))})
        log(f'Bootstrap {nuevo} - {base} listo')
    return pd.DataFrame(filas)


def segmentos(X, y, meta, a, b, preds):
    yv = y.iloc[a:b].to_numpy()
    fechas = pd.to_datetime(meta.timestamp.iloc[a:b].to_numpy(), unit='ms', utc=True)
    trimestre = fechas.tz_convert(None).to_period('Q').astype(str)
    # Volatilidad reciente: coeficiente de variación de las últimas 144 tasas (≈ 1 día), causal.
    cv = (X.fee_std_144 / (X.fee_media_144 + 1)).to_numpy()
    cortes = np.quantile(cv[:a], [1 / 3, 2 / 3])  # cortes fijados con train
    vol = np.select([cv[a:b] <= cortes[0], cv[a:b] <= cortes[1]], ['baja', 'media'], 'alta')
    filas_t, filas_v = [], []
    for nombre, (p, u) in preds.items():
        for etiqueta, mascara_por in [('trimestre', trimestre), ('volatilidad', vol)]:
            for seg in pd.unique(mascara_por):
                m = mascara_por == seg
                fila = {'conjunto': nombre, 'segmento': seg, **metricas_segmento(yv[m], p[m], p[m] >= u)}
                (filas_t if etiqueta == 'trimestre' else filas_v).append(fila)
    orden = {'baja': 0, 'media': 1, 'alta': 2}
    vt = pd.DataFrame(filas_t).sort_values(['conjunto', 'segmento'])
    vv = pd.DataFrame(filas_v).assign(o=lambda d: d.segmento.map(orden)).sort_values(['conjunto', 'o']).drop(columns='o')
    return vt, vv, cortes


def errores(X, y, meta, a, b, preds):
    raw, _ = experiment.load_source()
    tasa = raw.set_index('height').median_fee_rate
    h = meta.height.iloc[a:b].to_numpy()
    siguiente = tasa.reindex(h + 1).to_numpy()
    theta = X.umbral_local.iloc[a:b].to_numpy()
    margen = (siguiente + 1) / (theta + 1)  # misma convención +1 que fee_relativa
    yv = y.iloc[a:b].to_numpy()
    assert np.array_equal(yv, (siguiente > theta).astype(int))  # verificación del target
    prev = y.shift(1).iloc[a:b].to_numpy().astype(int)
    transicion = np.select([(prev == 0) & (yv == 0), (prev == 1) & (yv == 1), (prev == 0) & (yv == 1)],
                           ['0→0 baja estable', '1→1 alta estable', '0→1 inicio de alza'], '1→0 fin de alza')
    bins = [0, .5, .8, .95, 1.0, 1.05, 1.25, 2, np.inf]
    nombres = ['<0.5', '0.5–0.8', '0.8–0.95', '0.95–1', '1–1.05', '1.05–1.25', '1.25–2', '≥2']
    tramo = pd.cut(margen, bins=bins, labels=nombres, right=True)
    filas_m, filas_tr, filas_rach = [], [], []
    for nombre in ['C_todas', 'A_comisiones', 'persistencia']:
        p, u = preds[nombre]; pred = (p >= u).astype(int); err = pred != yv
        for seg in nombres:
            m = np.asarray(tramo == seg)
            filas_m.append({'conjunto': nombre, 'margen_siguiente_vs_umbral': seg, 'n': int(m.sum()),
                            'clase_real': int(yv[m].max()) if m.any() else np.nan,
                            'tasa_error': float(err[m].mean()) if m.any() else np.nan})
        for seg in ['0→0 baja estable', '1→1 alta estable', '0→1 inicio de alza', '1→0 fin de alza']:
            m = transicion == seg
            filas_tr.append({'conjunto': nombre, 'transicion': seg, 'n': int(m.sum()),
                             'tasa_error': float(err[m].mean())})
        filas_rach.append({'conjunto': nombre, 'tasa_error': float(err.mean()),
                           'p_error_dado_error_previo': float(err[1:][err[:-1]].mean()),
                           'falsos_positivos': int((pred & (yv == 0)).sum()),
                           'falsos_negativos': int(((1 - pred) & yv).sum())})
    # Perfil de FP y FN del modelo C frente a aciertos (medianas de variables clave).
    p, u = preds['C_todas']; pred = (p >= u).astype(int)
    tipo = np.select([(pred == 1) & (yv == 1), (pred == 0) & (yv == 0), (pred == 1) & (yv == 0)],
                     ['VP', 'VN', 'FP'], 'FN')
    Xv = X.iloc[a:b].assign(tipo=tipo, margen=margen,
                            cv_144=(X.fee_std_144 / (X.fee_media_144 + 1)).iloc[a:b].to_numpy())
    perfil = Xv.groupby('tipo')[['fee_relativa', 'margen', 'cv_144', 'median_fee_rate', 'tx_count',
                                 'size_relativo_1008', 'intervalo_segundos']].median()
    perfil.insert(0, 'n', Xv.tipo.value_counts())
    return (pd.DataFrame(filas_m), pd.DataFrame(filas_tr), pd.DataFrame(filas_rach),
            perfil.loc[['VP', 'VN', 'FP', 'FN']].reset_index())


# --- Figuras ----------------------------------------------------------------
def estilo(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e4e3df', lw=.8); ax.set_axisbelow(True)


def figuras(resumen, vt, vv, err_m, err_tr):
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11})
    orden = list(CONJUNTOS) + ['persistencia']
    # 1. Ablation: AP y F1 en validación
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    r = resumen.set_index('conjunto').loc[orden]
    # La persistencia solo emite 0/1: su AP no es comparable y se omite en ese panel.
    for ax, met, titulo, orden in [(axes[0], 'val_average_precision', 'Precisión promedio (AP)', list(CONJUNTOS)),
                                   (axes[1], 'val_f1', 'F1 de la clase alta', orden)]:
        vals = r.loc[orden, met].to_numpy()
        ax.bar(range(len(orden)), vals, color=[COLORES[c] for c in orden], width=.7,
               edgecolor='white', linewidth=2)
        for i, v in enumerate(vals):
            ax.text(i, v + .01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        ax.set_xticks(range(len(orden)), [ETIQUETAS[c] for c in orden], rotation=25, ha='right')
        ax.set_ylim(0, max(vals) * 1.18); ax.set_title(titulo); estilo(ax)
    fig.suptitle('Ablation — validación cronológica (mayo 2021 – agosto 2022)')
    fig.tight_layout(); fig.savefig(FIG / 'ablation_validacion.png', dpi=180); plt.close(fig)
    # 2. F1 por trimestre (pequeños múltiplos no necesarios: una sola métrica)
    fig, ax = plt.subplots(figsize=(8, 4))
    for c in ['A0_tasa_actual', 'A_comisiones', 'C_todas', 'persistencia']:
        d = vt[vt.conjunto == c]
        ax.plot(d.segmento, d.f1, marker='o', ms=6, lw=2, color=COLORES[c], label=ETIQUETAS[c],
                ls='--' if c == 'persistencia' else '-')
    ax.set(ylabel='F1 de la clase alta', ylim=(0, 1), title='Rendimiento por trimestre — validación')
    ax.legend(frameon=False, ncol=2, loc='upper left'); estilo(ax)
    fig.tight_layout(); fig.savefig(FIG / 'f1_por_trimestre.png', dpi=180); plt.close(fig)
    # 3. F1 por volatilidad reciente
    fig, ax = plt.subplots(figsize=(7, 4))
    cs = ['A0_tasa_actual', 'A_comisiones', 'C_todas', 'persistencia']; w = .2
    for j, c in enumerate(cs):
        d = vv[vv.conjunto == c]
        ax.bar(np.arange(3) + (j - 1.5) * w, d.f1, w, color=COLORES[c], label=ETIQUETAS[c],
               edgecolor='white', linewidth=1.5)
    ax.set_xticks(range(3), ['Baja', 'Media', 'Alta'])
    ax.set(xlabel='Volatilidad de las últimas 144 tasas (terciles fijados en train)',
           ylabel='F1 de la clase alta', ylim=(0, 1), title='Rendimiento según volatilidad reciente — validación')
    ax.legend(frameon=False, ncol=2, loc='upper left'); estilo(ax)
    fig.tight_layout(); fig.savefig(FIG / 'f1_por_volatilidad.png', dpi=180); plt.close(fig)
    # 4. Tasa de error según el margen del bloque siguiente respecto al umbral
    fig, ax = plt.subplots(figsize=(8, 4))
    d = err_m[err_m.conjunto == 'C_todas']
    col = ['#2a78d6' if c == 0 else '#eb6834' for c in d.clase_real]
    ax.bar(d.margen_siguiente_vs_umbral, d.tasa_error, color=col, edgecolor='white', linewidth=2)
    for i, (v, n) in enumerate(zip(d.tasa_error, d.n)):
        ax.text(i, v + .01, f'{v:.0%}\nn={n:,}'.replace(',', ' '), ha='center', va='bottom', fontsize=8)
    ax.axvline(3.5, color='#52514e', ls=':', lw=1)
    ax.text(3.45, .93, 'clase real 0', ha='right', fontsize=9, color='#52514e')
    ax.text(3.55, .93, 'clase real 1', ha='left', fontsize=9, color='#52514e')
    ax.set(xlabel='Tasa del bloque siguiente / umbral móvil', ylabel='Tasa de error', ylim=(0, 1.05),
           title='Errores del modelo C según la distancia al umbral — validación'); estilo(ax)
    fig.tight_layout(); fig.savefig(FIG / 'errores_por_margen.png', dpi=180); plt.close(fig)
    # 5. Tasa de error por tipo de transición
    fig, ax = plt.subplots(figsize=(8, 4))
    cats = ['0→0 baja estable', '1→1 alta estable', '0→1 inicio de alza', '1→0 fin de alza']; w = .27
    for j, c in enumerate(['A_comisiones', 'C_todas', 'persistencia']):
        d = err_tr[err_tr.conjunto == c].set_index('transicion').loc[cats]
        ax.bar(np.arange(4) + (j - 1) * w, d.tasa_error, w, color=COLORES[c], label=ETIQUETAS[c],
               edgecolor='white', linewidth=1.5)
    ax.set_xticks(range(4), cats)
    ax.set(ylabel='Tasa de error', ylim=(0, 1.05), title='Errores según la transición de clase — validación')
    ax.legend(frameon=False, loc='upper left'); estilo(ax)
    fig.tight_layout(); fig.savefig(FIG / 'errores_por_transicion.png', dpi=180); plt.close(fig)


def run():
    FIG.mkdir(parents=True, exist_ok=True)
    X, y, meta, a, b, audit = cargar_desarrollo()
    assert set(CONJUNTOS['C_todas']) == set(X.columns) and len(X.columns) == 33
    log(f'Desarrollo: {a} train + {b - a} validación. Test no utilizado.')
    folds, resumen, preds = ablation(X, y, a, b)
    folds.to_csv(OUT / 'ablation_folds.csv', index=False)
    resumen.to_csv(OUT / 'ablation_resumen.csv', index=False)
    pd.DataFrame({'height': meta.height.iloc[a:b].to_numpy(), 'y': y.iloc[a:b].to_numpy(),
                  **{f'p_{k}': v[0] for k, v in preds.items()}}).to_csv(
        OUT / 'predicciones_validacion_ablation.csv', index=False, float_format='%.6f')
    boot = bootstrap_diferencias(y.iloc[a:b], preds, [
        ('A_comisiones', 'A0_tasa_actual'), ('B_comisiones_actividad', 'A_comisiones'),
        ('C_todas', 'B_comisiones_actividad'), ('A0_tasa_actual', 'persistencia')])
    boot.to_csv(OUT / 'ablation_bootstrap.csv', index=False)
    vt, vv, cortes = segmentos(X, y, meta, a, b, preds)
    vt.to_csv(OUT / 'segmentos_trimestre.csv', index=False)
    vv.to_csv(OUT / 'segmentos_volatilidad.csv', index=False)
    err_m, err_tr, err_r, perfil = errores(X, y, meta, a, b, preds)
    err_m.to_csv(OUT / 'errores_margen.csv', index=False)
    err_tr.to_csv(OUT / 'errores_transicion.csv', index=False)
    err_r.to_csv(OUT / 'errores_resumen.csv', index=False)
    perfil.to_csv(OUT / 'errores_perfil_modeloC.csv', index=False)
    figuras(resumen, vt, vv, err_m, err_tr)
    manifest = {'test_evaluado': False, 'seed': SEED, 'gap': GAP, 'hgb': HGB,
                'conjuntos': CONJUNTOS, 'cortes_volatilidad_cv144_train': cortes.tolist(),
                'bootstrap': {'n': N_BOOT, 'bloque': BLOQUE},
                'python': platform.python_version(), 'numpy': np.__version__,
                'pandas': pd.__version__, 'scikit_learn': sklearn.__version__,
                'code_sha256': hashlib.sha256(open(__file__, 'rb').read()).hexdigest()}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    log('Terminado. Test no evaluado.')


if __name__ == '__main__':
    run()
