# Clasificación de comisiones por bloque de Bitcoin

Proyecto de Machine Learning (CS3061), UTEC, 2026-2.

El objetivo es clasificar la tasa mediana de comisión del siguiente bloque como alta o no alta, utilizando información de bloques anteriores.

## Estructura

```text
REPOSITORIO
├── README.md
├── .gitignore
├── docs/
│   ├── main.tex
│   └── utec_logo.png
└── eda/
    ├── dataset.csv
    ├── 01_eda.ipynb
    └── requirements.txt
```

## Dataset

Fuente: Jesús Graterol, [Bitcoin Blockchain Historical Data](https://www.kaggle.com/datasets/jesusgraterol/bitcoin-blockchain-dataset), Kaggle.

[eda/dataset.csv](eda/dataset.csv) contiene 810 909 bloques y 13 columnas, con registros hasta octubre de 2023. Cada fila representa un bloque. El archivo está incluido y se conserva sin modificaciones.

Las tasas se analizan en la escala publicada; su unidad y definición históricas exactas requieren verificación adicional con la fuente.

## EDA básico

[eda/01_eda.ipynb](eda/01_eda.ipynb) incluye:

- Carga del CSV, primeras filas, dimensiones y tipos.
- Valores faltantes, duplicados y revisión básica de valores.
- Estadísticas descriptivas con `describe()`.
- Histogramas y boxplots.
- Revisión de tasas cero y bloques con una transacción.
- Detección de valores extremos con la regla IQR, sin eliminarlos automáticamente.
- Definición de la clase y gráfico de su frecuencia.
- Correlación de Pearson y un diagrama de dispersión.
- Evolución anual de la tasa mediana, con una tabla del número de bloques por año.

Se conservan las particiones temporales de la propuesta: 70 % entrenamiento, 15 % validación y 15 % prueba. Los gráficos y el umbral se calculan solo con entrenamiento, después de las revisiones estructurales del archivo completo. No se entrenan modelos.

Las tablas y los siete gráficos están dentro del notebook. No se generan informes de conclusiones, carpetas de resultados ni archivos de registro. El gráfico anual usa solo entrenamiento y señala que 2019 está incompleto. Las características históricas y el modelado quedan para etapas posteriores.

### Ejecutar

Entorno comprobado: Python 3.13. Desde la raíz del repositorio:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r eda/requirements.txt
```

Abrir `eda/01_eda.ipynb` en VS Code/Jupyter, seleccionar el entorno `.venv` y ejecutar las celdas en orden. El directorio de trabajo debe ser `eda/`, porque el notebook carga directamente `pd.read_csv("dataset.csv")`.

El notebook ya contiene las salidas de una ejecución completa. No necesita internet para analizar el dataset.

## Propuesta

[docs/main.tex](docs/main.tex) contiene la propuesta. Para Overleaf, subir ese archivo y `docs/utec_logo.png` juntos. Completar los campos de integrantes, códigos y docente.

Para compilar localmente:

```sh
cd docs
pdflatex main.tex
pdflatex main.tex
```

El artículo P1 definitivo se adaptará al template IEEE exigido por el curso.

## Integrantes

1. Ordinola Ortega, Carlos David - 
2. Guerrero Gutierrez, Nayeli Belén - 202410790
3. Alvarado León, Adriana Celeste - 202420154

Repositorio: https://github.com/naye-gg/bitcoin-block-fee-classification_ML_project
