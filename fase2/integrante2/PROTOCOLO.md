# Protocolo experimental — integrante 2

Implementación local del plan P2; pendiente de adoptar por el equipo. No constituye un acuerdo ya confirmado por los otros integrantes.

## Fuente y decisiones heredadas de P1

- Repositorio: https://github.com/naye-gg/bitcoin-block-fee-classification_ML_project
- Rama `EDA_APARTE`, commit `7fbe2058c800eb5726f557b49e340bca4e338296`.
- Notebook: `bitcoin-fee-eda-recortado/eda/01_eda.ipynb`, conservado en `reference/`.
- Dataset original: 810 909 filas, 13 columnas. SHA-256 `408620117782d1eb68a38eda4c2b2f6cdfc23ccedd6999ca8740d222a835ac6e`.
- Orden por `height`; filtro por timestamp UTC >= 2016-01-01, antes de construir ventanas. Hay 419 728 bloques y 419 727 pares.
- Para el bloque h, theta_h = percentil 75 de las 1008 tasas desde h−1007 hasta h, ambos incluidos. `rolling(1008).quantile(.75)` usa la interpolación lineal de pandas.
- y_h = 1 si `median_fee_rate[h+1] > theta_h`; la igualdad corresponde a 0. El siguiente bloque solo interviene en la etiqueta.
- Eliminar las 1007 filas iniciales sin ventana completa. Quedan 418 720 ejemplos. Aplicar `int(n*.70)` y `int(n*.85)` **después** de esa eliminación.
- Particiones nominales: 293 104 train; 62 808 validación; 62 808 test. Los límites por altura y fecha quedan en `results/particiones.csv`. No reordenar por timestamp.

## Características concretadas para P2

Esta selección desarrolla las propuestas de P1; el equipo debe usarla también en baselines si quiere aislar el efecto del algoritmo. `src/experiment.py::causal_features` es la definición reproducible y `results/features.csv` su lista de 33 columnas.

1. Once columnas actuales: size, tx_count, difficulty, median_fee_rate, avg_fee_rate, total_fees, fee_range_min, fee_range_max, input_count, output_count y output_amount.
2. Umbral móvil actual; razón `(median_fee_rate+1)/(umbral_local+1)`; razón `(avg_fee_rate+1)/(median_fee_rate+1)`. El +1 es una decisión numérica declarada en la escala publicada, no una unidad física verificada. La variable `ratio_avg_mediana` conserva esta convención suavizada.
3. Rezagos de la tasa mediana: 1, 6 y 144 bloques. Media y desviación poblacional (`ddof=0`) de 6, 144 y 1008 bloques, incluyendo h.
4. Tamaño, entradas y salidas por transacción; tamaño actual dividido por el máximo de los últimos 1008 bloques. Esa razón no es utilización exacta de capacidad ni peso de bloque.
5. Intervalo entre timestamps en segundos; los negativos pasan a NaN y se añade una bandera explícita. No se eliminan filas ni se cambia el orden de la cadena.
6. Hora UTC (incluye minutos) y día de semana representados por seno y coseno.

No se incluyen height, timestamp crudo, tasa_siguiente, etiqueta ni campos del bloque futuro. No se eliminan extremos por IQR. Un `SimpleImputer(strategy='median')` se ajusta exclusivamente en cada train. No se escalan las variables de estos árboles. Ningún resultado depende de un scaler aprendido con validación.

## Selección y comparación

- Semilla 42, tres folds de ventana **expansiva**, con 40 000 ejemplos de validación interna por fold y `gap=1008`. Fronteras exactas en `results/folds.csv`.
- Se conserva el gap conservador propuesto en P1. Compartir historia pasada entre ejemplos no constituye por sí solo fuga; este margen es una decisión de diseño adicional y reduce la cantidad de datos de ajuste.
- Tres candidatos por familia (18 ajustes en total). Se elige mayor F1 medio de CV con umbral 0.5; empate: primer candidato. No hay búsqueda posterior basada en validación externa.
- Random Forest: 100 árboles; `max_features='sqrt'`, bootstrap con `max_samples=.5`; tres pares (profundidad, mínimo de hoja): (12,30), (20,30), (sin límite,80). Sin pesos de clase.
- HistGradientBoosting: (iteraciones, hojas, learning rate, mínimo hoja, L2): (120,15,.08,40,1), (180,31,.06,80,5), (180,15,.10,80,5). `early_stopping=False`: se evita el holdout aleatorio interno. Número de iteraciones seleccionado temporalmente; sin pesos de clase.
- Para el candidato elegido, ajustar el umbral de probabilidad en una malla predefinida 0.10–0.90, paso .025, usando sus probabilidades fuera de fold (OOF). Maximizar F1 agrupado; empate: más cercano a .5. No confundir este umbral con theta_h, que define el target.
- Los resultados CV/OOF participan en selección y **no** son una estimación independiente del modelo elegido. No se presenta la desviación entre folds como intervalo de confianza.
- Reajustar cada ganador con las primeras 292 096 filas de train: las últimas 1008 forman el gap frente a validación. Evaluar una vez ambos en las mismas 62 808 filas de validación. Conservar la partición nominal para integrarse con P1.
- Train se reporta sobre esas mismas 292 096 filas de ajuste; su brecha frente a validación puede reflejar tanto sobreajuste como cambios temporales. No atribuirla exclusivamente a uno.

## Métricas e interpretación

F1 de clase alta es la métrica principal. Se exportan precision, recall, `average_precision` (AP), ROC-AUC, matriz TN/FP/FN/TP y también PR-AUC trapezoidal con su nombre explícito. **AP y área trapezoidal PR son cálculos distintos**; la tabla principal usa AP. Curvas y AUC reciben probabilidades, no etiquetas binarias.

Se utiliza la importancia por reducción de impureza (`feature_importances_`) de RF, como en la sesión 09b, sección 10. Es un diagnóstico descriptivo, susceptible a sesgo por cardinalidad y correlación; no demuestra causalidad. HGB no ofrece esa propiedad, por lo que se interpreta mediante curvas, métricas y errores. Se retiró la extensión propia de permutaciones temporales para mantener la entrega dentro del enfoque del curso.


## Test e integración

El script no construye `y_test`, no predice test y no calcula su balance. El CSV fuente se lee completo para reproducir P1 y se guardan metadatos de las fronteras ya establecidas. La construcción causal de características no aprende estadísticas globales.

Antes de integrar: compartir este protocolo, comprobar hash, target, índices, características y gap con los otros dos integrantes. Añadir las métricas de persistencia y logística cuando estén disponibles; no inventarlas ni afirmar que los ensambles las superan. El P1 ya inspeccionó distribuciones de test para reformular el problema: debe declararse esa exposición histórica, aunque no se use test en esta fase de modelado.

La evaluación final en test queda pendiente de congelar el protocolo del equipo, los modelos, los umbrales y la política de reajuste. No ejecutar búsquedas ni cambiar configuraciones después de observar ese resultado.
