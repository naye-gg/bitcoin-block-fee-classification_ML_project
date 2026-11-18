# P2 — Modelos comparativos de Nayeli (integrante 2)

Random Forest y HistGradientBoosting para clasificar el nivel relativo de comisión del siguiente bloque de Bitcoin. Se reproduce el objetivo móvil de P1, rama `EDA_APARTE`. El test queda reservado.

## Archivos

- `notebooks/02_models.ipynb`: explicación, ejecución y resultados de tu parte.
- `src/experiment.py`: preparación causal de datos, búsqueda temporal, entrenamiento, evaluación y figuras.
- `PROTOCOLO.md`: definición exacta para compartir con los integrantes 1 y 3.
- `results/`: métricas CSV, hiperparámetros, probabilidades de validación y gráficos.
- `paper/metodologia_modelos.tex` y `paper/resultados_modelos.tex`: fragmentos para insertar en el paper IEEE.
- `reference/01_eda_recortado_P1.ipynb`: copia de la fuente, sin modificaciones.

## Datos

En este repositorio el CSV ya está en `../../eda/dataset.csv` desde la raíz de esta entrega. Copiarlo a `data/dataset.csv` antes de ejecutar: `cp ../../eda/dataset.csv data/dataset.csv`. Para otra máquina, obtener el archivo usado en P1 de [la fuente](https://www.kaggle.com/datasets/jesusgraterol/bitcoin-blockchain-dataset) o del repositorio del equipo y colocarlo en esa ruta. Una versión actualizada de Kaggle puede ser distinta: el script exige el SHA-256 documentado en `PROTOCOLO.md`. No subir automáticamente el CSV de 70 MB; se incluye en `.gitignore`.

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

Esta entrega contiene tus dos familias de modelos, búsqueda, métricas, curvas PR, matrices e interpretación. Las características derivadas son una concreción de las propuestas de P1: compartirlas con el equipo antes de consolidar comparaciones. No contiene resultados de los modelos del integrante 1 ni el ablation del integrante 3. Las cifras son de validación, no del test final. La validación podrá participar en la integración; por ello no reemplaza la evaluación final reservada.
