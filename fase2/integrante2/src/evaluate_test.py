"""Evaluación final con selección congelada y sin reajuste sobre validación."""
import json
import hashlib
import platform
from datetime import datetime, timezone
from pathlib import Path

from experiment import (
    ROOT, GAP, SEED, WINDOW, load_source, causal_features, estimator, metrics,
    np, pd, sklearn, threadpool_limits, log,
)


def run():
    out = ROOT / 'results' / 'test_final'
    out.mkdir(exist_ok=False)  # No sobrescribir una evaluación ya realizada.
    selection_path = ROOT / 'results/seleccion.json'
    choices = json.loads(selection_path.read_text())
    original = json.loads((ROOT / 'results/manifest.json').read_text())
    assert choices == original['seleccion']
    assert sklearn.__version__ == original['scikit_learn']
    df, sha = load_source(ROOT.parents[1] / 'eda/dataset.csv')
    current = df.iloc[:-1]
    theta = current.median_fee_rate.rolling(WINDOW).quantile(.75)
    valid = theta.notna()
    X = causal_features(current).loc[valid].reset_index(drop=True)
    meta = current.loc[valid, ['height', 'timestamp']].reset_index(drop=True)
    n = len(X)
    a, b = int(n * .70), int(n * .85)
    fit_end = a - GAP
    assert (n, a, b, fit_end) == (418720, 293104, 355912, 292096)
    assert list(X.columns) == original['features']
    future = df.median_fee_rate.shift(-1).iloc[:-1].loc[valid].reset_index(drop=True)
    thresholds = theta.loc[valid].reset_index(drop=True)
    yfit = (future.iloc[:fit_end] > thresholds.iloc[:fit_end]).astype('int8')
    protocol = {
        'fecha_utc': datetime.now(timezone.utc).isoformat(),
        'politica': 'Reproducir ajuste original de train; no reajustar con validación ni seleccionar con test.',
        'dataset_sha256': sha, 'seleccion': choices, 'seed': SEED,
        'fit_rows': fit_end, 'test_inicio_iloc': b, 'test_filas': n-b,
        'test_height_inicial': int(meta.height.iloc[b]),
        'test_height_final': int(meta.height.iloc[-1]),
        'python': platform.python_version(), 'numpy': np.__version__,
        'pandas': pd.__version__, 'scikit_learn': sklearn.__version__,
        'seleccion_sha256': hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'experiment_sha256': hashlib.sha256((ROOT/'src/experiment.py').read_bytes()).hexdigest(),
        'test_evaluado': False,
    }
    manifest = out / 'manifest.json'
    manifest.write_text(json.dumps(protocol, indent=2, ensure_ascii=False))
    models = {}
    for family, choice in choices.items():
        log('Reproduciendo ajuste original: ' + family)
        model = estimator(family, choice['params'])
        with threadpool_limits(limits=4):
            model.fit(X.iloc[:fit_end], yfit)
            pval = model.predict_proba(X.iloc[a:b])[:, 1]
        saved = pd.read_csv(ROOT / f'results/predicciones_validacion_{family}.csv')
        np.testing.assert_array_equal(saved.height, meta.height.iloc[a:b])
        np.testing.assert_array_equal(saved.y, (future.iloc[a:b] > thresholds.iloc[a:b]).astype(int))
        np.testing.assert_allclose(saved.probabilidad, pval, rtol=0, atol=1e-12)
        models[family] = model
        log('Predicciones de validación reproducidas: ' + family)
    # Test solo se evalúa tras verificar ambos modelos congelados.
    ytest = (future.iloc[b:] > thresholds.iloc[b:]).astype('int8')
    rows = []
    for family, model in models.items():
        threshold = choices[family]['threshold']
        with threadpool_limits(limits=4):
            p = model.predict_proba(X.iloc[b:])[:, 1]
        rows.append({'modelo': family, 'particion': 'prueba', **metrics(ytest, p, threshold)})
        pd.DataFrame({'height': meta.height.iloc[b:].to_numpy(), 'y': ytest.to_numpy(),
                      'probabilidad': p, 'prediccion': (p >= threshold).astype(int)}).to_csv(
                          out / f'predicciones_{family}.csv', index=False)
    table = pd.DataFrame(rows)
    table.to_csv(out / 'metricas.csv', index=False)
    persistence = (current.loc[valid, 'median_fee_rate'].reset_index(drop=True).iloc[b:] > thresholds.iloc[b:]).astype(int)
    baseline = metrics(ytest, persistence.to_numpy(), .5)
    expected = pd.read_csv(ROOT.parent / 'integrante1/results/baselines_resultados.csv')
    reference = expected.loc[(expected.modelo == 'persistencia') & (expected.particion == 'prueba')].iloc[0]
    for key in ['f1', 'precision', 'recall']:
        np.testing.assert_allclose(baseline[key], reference[key], rtol=0, atol=1e-12)
    protocol.update(test_evaluado=True, validacion_reproducida=True,
                    persistencia_verificada=True, positivos_test=int(ytest.sum()))
    manifest.write_text(json.dumps(protocol, indent=2, ensure_ascii=False))
    print(table.to_string(index=False), flush=True)


if __name__ == '__main__':
    run()
