# Predicción del nivel de comisiones por bloque en la red Bitcoin

Proyecto Final de Machine Learning (CS3061) — UTEC, 2026-2
Etapa 1: formulación del problema, descripción del dataset y análisis exploratorio.

## El problema

Cuando una persona emite una transacción en Bitcoin debe elegir cuánta comisión ofrece a los mineros. Si ofrece poco, la transacción puede tardar horas en confirmarse; si ofrece de más, paga un sobrecosto que no puede recuperar, porque las transacciones son irreversibles.

El espacio en un bloque es limitado y los mineros seleccionan qué transacciones incluyen. Cuando la demanda supera esa capacidad se forma una subasta, y el precio resultante varía de forma considerable en cuestión de horas. Las carteras incorporan estimadores de comisión, pero su desempeño es limitado justamente en los episodios de mayor congestión.

## Objetivo

Determinar si, a partir del estado observable de la red hasta el bloque actual, es posible anticipar si la comisión del siguiente bloque será alta respecto al nivel vigente del mercado.

El resultado no constituye un estimador de comisión completo, sino su primer componente: la estimación del nivel de precio del mercado, sobre la cual un estimador operativo aplicaría después el criterio de urgencia de cada usuario.

## Formulación como problema de aprendizaje (E, T, P)

| | |
|---|---|
| **Tarea (T)** | Clasificación binaria: dado el bloque *t* y su historia, predecir si la tasa mediana de comisión del bloque *t+1* superará el percentil 75 de los bloques recientes |
| **Experiencia (E)** | 810 908 pares (bloque, bloque siguiente) del registro de la cadena entre enero de 2009 y octubre de 2023, con trece atributos observables por bloque |
| **Desempeño (P)** | F1 sobre la clase «alta» y PR-AUC, sobre una partición temporal posterior a la de entrenamiento, comparados contra una referencia de persistencia |

No se emplea la exactitud como métrica principal: con clases desbalanceadas, un clasificador que prediga siempre la clase mayoritaria obtiene valores altos sin capacidad predictiva.

## Alcance

- **Se predice el nivel de comisión, no la ocupación del bloque.** La ocupación es la condición que da origen al mercado de comisiones y se analiza con ese propósito, pero no es la variable objetivo.
- **Se modela el mercado agregado, no la decisión individual.** La unidad de observación es el bloque, que resume cientos o miles de transacciones.
- **Los datos describen transacciones confirmadas.** Se observa el resultado de la subasta, no el conjunto de ofertas.
- **No se dispone del estado del mempool**, que no puede reconstruirse retrospectivamente.

## Preguntas de investigación

1. ¿La tasa de comisión del bloque actual permite anticipar la del siguiente por encima de lo que logra una regla de persistencia?
2. ¿Qué atributos del estado de la red aportan información adicional a la tasa reciente?
3. ¿El desempeño del clasificador se mantiene estable a lo largo del periodo, o varía entre tramos con distinto nivel de congestión?

## Hipótesis

- **H1.** La tasa de comisión reciente es el predictor dominante.
- **H2.** Los atributos de actividad de la red aportan una mejora marginal sobre H1, menor que la contribución de la propia tasa.
- **H3.** El desempeño es menor en los tramos del periodo donde la comisión no se determinaba por competencia entre usuarios.

## Conjunto de datos

Fuente: Jesús Graterol, [Bitcoin Blockchain Historical Data](https://www.kaggle.com/datasets/jesusgraterol/bitcoin-blockchain-dataset), Kaggle.

810 909 bloques y 13 columnas, de enero de 2009 a octubre de 2023. Cada fila representa un bloque. Se emplea el registro completo, sin restringir el periodo: el volumen y la diversidad de condiciones de red resultan de interés para evaluar la estabilidad del comportamiento a lo largo del tiempo.

La variable de interés es `median_fee_rate`: la mediana de las tasas de comisión pagadas por las transacciones incluidas en cada bloque, en satoshis por byte. Se interpreta como el precio aproximado que era necesario ofrecer para ser incluido en ese bloque. El objetivo se construye desplazando esa columna un bloque hacia adelante.

**El CSV no está incluido en este paquete por su tamaño (70 MB).** Descargarlo de Kaggle y colocarlo en `eda/dataset.csv` antes de ejecutar el notebook.

## Estructura del repositorio

```text
├── README.md
├── .gitignore
├── docs/
│   ├── main.tex
│   └── utec_logo.png
└── eda/
    ├── dataset.csv          <- colocar aquí el CSV descargado de Kaggle
    ├── 01_eda.ipynb
    └── requirements.txt
```

## Contenido del notebook

`eda/01_eda.ipynb` desarrolla, en veinte secciones: formulación del problema y del aprendizaje (1-7); descripción del conjunto de datos y de sus variables (8); calidad de los datos, incluida la coherencia interna de las tasas (9); construcción del objetivo y partición temporal (10); estadísticas descriptivas, distribuciones, tasas cero y valores extremos (11-14); definición y validación de la variable objetivo (15); relaciones entre variables (16); evolución temporal y condiciones de formación del precio (17); prevención de fuga de datos (18); siguientes pasos (19) y limitaciones (20).

## Hallazgos principales

**1. Un umbral fijo produce una partición de prueba no evaluable.**

| Partición | Bloques | Periodo | Mediana de tasa | % clase alta |
|---|---|---|---|---|
| Entrenamiento | 567 635 | 2009-01-09 a 2019-03-18 | 17 | 24,98 % |
| Validación | 121 636 | 2019-03-18 a 2021-07-01 | 24 | 31,25 % |
| Prueba | 121 637 | 2021-07-01 a 2023-10-06 | 8 | **2,31 %** |

Con 2,31 % de positivos, un clasificador que prediga siempre «no alta» obtiene 97,7 % de exactitud sin capacidad predictiva.

**2. Un umbral móvil corrige el desbalance.**

Definiendo la clase contra el percentil 75 de los 1008 bloques anteriores —una semana aproximadamente— en lugar de un valor constante, la proporción pasa a 17,40 %, 26,07 % y 23,18 %. El resultado es estable para ventanas de un día, una semana y un mes.

La clase pasa a significar «tasa alta respecto al nivel de la última semana», que es la comparación relevante para decidir si conviene emitir una transacción ahora o esperar.

**3. La correlación agregada de 0,05 resulta de mezclar épocas.**

| Época | Pearson | Spearman | Pearson sobre log |
|---|---|---|---|
| 2009-2011 | 0,042 | 0,254 | 0,208 |
| 2012-2016 | 0,049 | 0,354 | 0,212 |
| 2017-2019 | **0,819** | 0,740 | 0,742 |
| 2020-2023 | **0,817** | 0,692 | 0,718 |
| Entrenamiento completo | 0,050 | 0,629 | 0,610 |

Dentro de cada época reciente la tasa del bloque actual es un predictor fuerte del siguiente, lo que sustenta H1 y fija una referencia exigente para P.

**4. Las condiciones de formación del precio cambian entre 2015 y 2016.**

| Año | % bloques >900 KB | % tasa cero | Correlación con el siguiente |
|---|---|---|---|
| 2014 | 1,1 % | 4,5 % | 0,090 |
| 2015 | 16,8 % | 6,4 % | 0,124 |
| 2016 | 62,8 % | 2,1 % | 0,205 |
| 2017 | 89,3 % | 1,0 % | 0,702 |
| 2023 | 96,2 % | 0,3 % | 0,911 |

Existieron comisiones en Bitcoin desde 2009, pero la comisión solo se determina por competencia cuando el espacio es escaso. El periodo analizado contiene por tanto dos mecanismos distintos, lo que se recoge como limitación y motiva la hipótesis H3.

## Decisiones metodológicas

- Partición temporal 70/15/15, sin mezclar el orden de los bloques.
- El umbral se estima solo con entrenamiento.
- Los diagnósticos de balance y coherencia usan el archivo completo: son revisiones estructurales del diseño experimental, no estiman parámetros ni seleccionan configuración. Se declara en la sección 18.
- No se entrenan modelos. Las variables derivadas y el modelado corresponden a la etapa 2.

## Ejecutar

Entorno comprobado: Python 3.13.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r eda/requirements.txt
```

Abrir `eda/01_eda.ipynb` en VS Code o Jupyter, seleccionar el entorno `.venv` y ejecutar las celdas en orden. El directorio de trabajo debe ser `eda/`, porque el notebook carga directamente `pd.read_csv("dataset.csv")`.

El notebook ya contiene las salidas de una ejecución completa. No necesita internet.

## Propuesta

`docs/main.tex` contiene la propuesta. Para Overleaf, subir ese archivo y `docs/utec_logo.png` juntos.

```sh
cd docs
pdflatex main.tex
pdflatex main.tex
```

**Pendiente:** actualizar el documento con la formulación E/T/P y la definición móvil de la clase, y migrarlo al template IEEE exigido por el curso.

## Referencias

- Jesús Graterol. [Bitcoin Blockchain Historical Data](https://www.kaggle.com/datasets/jesusgraterol/bitcoin-blockchain-dataset), Kaggle.
- Jesús Graterol. [Generador y esquema del dataset](https://github.com/jesusgraterol/bitcoin-blockchain-dataset-builder).
- M. Möser y R. Böhme. «Trends, Tips, Tolls: A Longitudinal Study of Bitcoin Transaction Fees». *Financial Cryptography and Data Security*, 2nd Workshop on BITCOIN Research, LNCS 8976, Springer, 2015, pp. 19-33. DOI: 10.1007/978-3-662-48051-9_2
- D. Easley, M. O'Hara y S. Basu. «From Mining to Markets: The Evolution of Bitcoin Transaction Fees». *Journal of Financial Economics* 134(1), 2019, pp. 91-109. DOI: 10.1016/j.jfineco.2019.03.004
- G. Huberman, J. D. Leshno y C. Moallemi. «Monopoly without a Monopolist: An Economic Analysis of the Bitcoin Payment System». *The Review of Economic Studies* 88(6), 2021, pp. 3011-3040. DOI: 10.1093/restud/rdab014
- Bitcoin Developer Documentation. [Block Chain](https://developer.bitcoin.org/reference/block_chain.html).

## Integrantes

1. Nombre y código por completar.
2. Guerrero Gutierrez, Nayeli Belén - 202410790
3. Nombre y código por completar.

Repositorio: https://github.com/naye-gg/bitcoin-block-fee-classification_ML_project
