# P2 — Modelos comparativos de Nayeli (integrante 2)

Random Forest y HistGradientBoosting para clasificar el nivel relativo de comisión del siguiente bloque de Bitcoin. Se reproduce el objetivo móvil de P1, rama `EDA_APARTE`. La evaluación final de test se documenta por separado en `results/test_final/`.

## Archivos

- `notebooks/02_models.ipynb`: procedimiento y resultados de validación de RF y HGB.
- `src/experiment.py`: preparación causal de datos, búsqueda temporal, entrenamiento, evaluación y figuras.
- `PROTOCOLO.md`: definición del objetivo, las variables y el procedimiento experimental.
- `results/`: métricas CSV, hiperparámetros, probabilidades de validación y gráficos.
- `paper/metodologia_modelos.tex` y `paper/resultados_modelos.tex`: fragmentos para insertar en el paper IEEE.
- `reference/01_eda_recortado_P1.ipynb`: copia de la fuente, sin modificaciones.

## Datos

En este repositorio el CSV ya está en `../../eda/dataset.csv` desde la raíz de esta entrega. Copiarlo a `data/dataset.csv` antes de ejecutar: `cp ../../eda/dataset.csv data/dataset.csv`. Para otra máquina, obtener el archivo usado en P1 de [la fuente](https://www.kaggle.com/datasets/jesusgraterol/bitcoin-blockchain-dataset) o del repositorio del equipo y colocarlo en esa ruta. Una versión actualizada de Kaggle puede ser distinta: el script exige el SHA-256 documentado en `PROTOCOLO.md`. La copia local del CSV está excluida del control de versiones mediante `.gitignore`.

## Ejecutar

Desde esta carpeta:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/experiment.py
```

La búsqueda ejecuta 18 ajustes y dos reajustes finales; puede tardar varios minutos según el equipo. Usa hasta cuatro hilos. Sobrescribe los resultados de esta carpeta, no los archivos originales de P1.

También se puede abrir `notebooks/02_models.ipynb` en Jupyter o VS Code. Si hay resultados completos, el notebook los muestra; cambiar `REENTRENAR = True` para repetir el experimento. En un directorio sin resultados los genera automáticamente. Si modificas configuración, características o código, vuelve a entrenar antes de utilizar las cifras del paper.

El experimento genera un manifiesto con versiones y configuración. Se verifican las particiones, el hash, la ausencia de variables futuras y la causalidad por invariancia de prefijos (`python src/check_protocol.py`).

## Alcance de la entrega

Esta carpeta contiene la búsqueda temporal, las métricas, las curvas PR, las matrices de confusión y la interpretación de RF y HGB. El notebook conserva el experimento de validación; la evaluación final de prueba está en `results/test_final/`. La comparación con los baselines y el análisis de ablación se presentan en el README general y en `../../paper/main.tex`. La logística utiliza 22 variables y los árboles 33, por lo que sus diferencias de rendimiento no pueden atribuirse únicamente al algoritmo.


## Evaluación final de prueba

`src/evaluate_test.py` reproduce los modelos seleccionados sobre el entrenamiento original, comprueba sus predicciones de validación y evalúa test con umbrales congelados (RF 0.300, HGB 0.350). No reajusta con validación. Lee directamente `../../eda/dataset.csv`.

Las métricas están en `results/test_final/metricas.csv`; el manifiesto y las predicciones de cada modelo permiten auditar el resultado. El notebook y `results/model_comparison.csv` conservan la evaluación original de desarrollo.

Para reproducir en una copia de trabajo sin `results/test_final/`, ejecutar `python src/evaluate_test.py`. El script rechaza sobrescribir una evaluación existente. No usar test para cambiar la selección ni ajustar umbrales.
