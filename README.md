# Clasificación de comisiones por bloque de Bitcoin

Proyecto de Machine Learning (CS3061), UTEC, 2026-2.

El objetivo es clasificar el siguiente bloque según si su tasa mediana de comisión será alta o no alta. Se utilizará información de bloques anteriores y un umbral calculado solo con entrenamiento.

## Archivos

- `main.tex`: propuesta en LaTeX, con portada e índice.
- `utec_logo.png`: logo utilizado en la portada.
- `eda/`: carpeta para el notebook y las figuras del análisis exploratorio. Todavía no se ha realizado el EDA completo.

## Propuesta

Subir `main.tex` y `utec_logo.png` juntos a Overleaf y compilar con pdfLaTeX. Completar los nombres, códigos y docente en los campos entre corchetes.

Para compilar localmente, ejecutar dos veces desde esta carpeta:

```sh
pdflatex main.tex
```

## Datos y EDA

Fuente: Jesús Graterol, [Bitcoin Blockchain Historical Data](https://www.kaggle.com/datasets/jesusgraterol/bitcoin-blockchain-dataset), Kaggle.

La copia utilizada contiene 810 909 bloques y 13 columnas. El CSV queda fuera de esta carpeta, en `../dataset.csv`; no se sube al repositorio. Quien clone el repositorio puede descargarlo de Kaggle y colocarlo en esa ruta.

El futuro notebook se guardará en `eda/01_eda.ipynb`. Si se ejecuta desde `eda/`, el CSV local se encuentra en `../../dataset.csv`.

El EDA analizará calidad de datos, distribuciones, evolución temporal, desbalance y relaciones entre variables. El artículo P1 definitivo se adaptará al template IEEE exigido por el curso.

## Integrantes

1. Nombre y código por completar.
2. Nombre y código por completar.
3. Nombre y código por completar.

Repositorio: https://github.com/naye-gg/bitcoin-block-fee-classification_ML_project
