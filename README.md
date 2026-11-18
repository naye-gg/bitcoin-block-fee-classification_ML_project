# Predicción del nivel relativo de comisiones del siguiente bloque de Bitcoin

Proyecto final de Machine Learning (CS3061), UTEC, 2026-2.

Clasificación binaria a nivel de bloque: anticipar si la tasa mediana de comisión del **bloque siguiente** superará el percentil 75 de las 1008 tasas más recientes (aproximadamente una semana), usando solo información disponible hasta el bloque actual.

## Integrantes

| Integrante | Código | Parte en la fase 2 |
|---|---|---|
| Ordinola Ortega, Carlos David | | 1 — EDA orientado a la predicción, persistencia y regresión logística |
| Guerrero Gutierrez, Nayeli Belén | 202410790 | 2 — Random Forest, HistGradientBoosting y validación temporal |
| Alvarado León, Adriana Celeste | 202420154 | 3 — Ablation, robustez, análisis de errores, paper y presentación |

## Resultados principales

| Método | F1 validación | AP validación | F1 prueba | AP prueba |
|---|---|---|---|---|
| Persistencia | 0.540 | — | 0.576 | — |
| Regresión logística | 0.557 | 0.593 | 0.570 | 0.617 |
| Random Forest | 0.564 | 0.623 | 0.605 | 0.684 |
| HistGradientBoosting | 0.571 | 0.630 | 0.609 | 0.690 |

- En prueba, RF (F1 0.6048) y HGB (0.6093) superan a la persistencia (0.5763); la logística (0.5704) no la supera. Son diferencias descriptivas, sin prueba de significancia.
- El ablation muestra que la ganancia viene de la historia reciente de las comisiones (+0.084 de AP); la actividad de la red aporta poco (+0.012) y el calendario nada.
- Los modelos fallan sobre todo al inicio de los episodios de comisión alta, que duran unos dos bloques. Falta el estado del *mempool*, que el dataset no contiene.

La prueba de los árboles conserva los modelos y umbrales originales (RF 0.300; HGB 0.350), sin reajuste con validación. Se reprodujeron las predicciones de validación y las métricas de persistencia antes de consolidar la comparación. [Métricas de prueba](fase2/integrante2/results/test_final/metricas.csv) y [protocolo](fase2/integrante2/PROTOCOLO.md).

## Estructura

```text
├── README.md
├── docs/                       propuesta inicial (P1)
├── eda/
│   ├── 01_eda.ipynb            EDA de P1 (versión recortada desde 2016)
│   ├── dataset.csv             dataset de P1
│   └── requirements.txt
├── fase2/
│   ├── integrante1/            baselines: notebooks/01_baselines.ipynb, resultados y texto
│   ├── integrante2/            ensambles: PROTOCOLO.md, src/experiment.py, notebooks/02_models.ipynb
│   └── integrante3/            robustez: src/robustez.py, notebooks/04_robustez.ipynb
├── paper/
│   ├── main.tex                artículo final (plantilla IEEE de conferencias)
│   └── figures/
└── presentation/
    └── slides.tex              presentación de 10 minutos (Beamer)
```

Cada carpeta de `fase2/` tiene su propio README con instrucciones y detalle de resultados.

## Dataset

Fuente: Jesús Graterol, [Bitcoin Blockchain Historical Data](https://www.kaggle.com/datasets/jesusgraterol/bitcoin-blockchain-dataset), Kaggle. 810 909 bloques y 13 columnas, de enero de 2009 a octubre de 2023.

El archivo usado está en [eda/dataset.csv](eda/dataset.csv). Su SHA-256 es `408620117782d1eb68a38eda4c2b2f6cdfc23ccedd6999ca8740d222a835ac6e`. En Windows con `core.autocrlf=true`, la copia de trabajo puede tener saltos de línea CRLF y otro hash; para obtener el archivo exacto:

```bash
git show HEAD:eda/dataset.csv > fase2/integrante2/data/dataset.csv
git show HEAD:eda/dataset.csv > fase2/integrante3/data/dataset.csv
```

## Protocolo común

- Bloques ordenados por `height`, desde el 1 de enero de 2016; 418 720 ejemplos tras descartar las 1007 filas sin ventana completa.
- Partición cronológica 70/15/15: 293 104 / 62 808 / 62 808 ejemplos, con 24.65 % / 21.47 % / 24.93 % de positivos.
- Variables construidas solo con información hasta el bloque actual; transformaciones ajustadas en entrenamiento.
- Métrica principal: F1 de la clase alta. Secundarias: precisión, exhaustividad, AP y ROC-AUC.
- Detalle completo en [fase2/integrante2/PROTOCOLO.md](fase2/integrante2/PROTOCOLO.md).

## Reproducir

Python 3.12 o 3.13 (en 3.14 no hay wheel de `numpy==2.2.6`).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows (Git Bash): source .venv/Scripts/activate
```

| Parte | Dependencias | Ejecución |
|---|---|---|
| EDA (P1) | `eda/requirements.txt` | `eda/01_eda.ipynb`, con `eda/` como directorio de trabajo |
| Integrante 1 | `fase2/integrante1/requirements.txt` | `fase2/integrante1/notebooks/01_baselines.ipynb` |
| Integrante 2 | `fase2/integrante2/requirements.txt` | `python src/experiment.py` desde `fase2/integrante2/` |
| Integrante 3 | `fase2/integrante3/requirements.txt` | `python src/robustez.py` desde `fase2/integrante3/` |

Los notebooks incluyen las salidas de una ejecución completa.

## Paper y presentación

`paper/main.tex` y `presentation/slides.tex` compilan en Overleaf: subir cada carpeta completa (con `figures/` o `utec_logo.png`) y compilar con pdfLaTeX. Localmente:

```bash
cd paper && pdflatex main.tex && pdflatex main.tex
```

Repositorio: https://github.com/naye-gg/bitcoin-block-fee-classification_ML_project
