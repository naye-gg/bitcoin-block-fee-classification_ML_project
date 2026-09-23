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
| **Experiencia (E)** | 419 727 pares (bloque, bloque siguiente) del registro de la cadena entre enero de 2016 y octubre de 2023, con trece atributos observables por bloque |
| **Desempeño (P)** | F1 sobre la clase «alta» y PR-AUC, sobre una partición temporal posterior a la de entrenamiento, comparados contra una referencia de persistencia |

No se emplea la exactitud como métrica principal: con clases desbalanceadas, un clasificador que prediga siempre la clase mayoritaria obtiene valores altos sin capacidad predictiva.

## Alcance

- **Se predice el nivel de comisión, no la ocupación del bloque.** La ocupación es la condición que da origen al mercado de comisiones y se analiza con ese propósito, pero no es la variable objetivo. Dentro del periodo estudiado los bloques operan cerca de su capacidad de forma habitual, por lo que su llenado no es un evento informativo.
- **Se modela el mercado agregado, no la decisión individual.** La unidad de observación es el bloque, que resume cientos o miles de transacciones.
- **Los datos describen transacciones confirmadas.** Se observa el resultado de la subasta, no el conjunto de ofertas.
- **No se dispone del estado del mempool**, que no puede reconstruirse retrospectivamente.

## Preguntas de investigación

1. ¿La tasa de comisión del bloque actual permite anticipar la del siguiente por encima de lo que logra una regla de persistencia?
2. ¿Qué atributos del estado de la red aportan información adicional a la tasa reciente?
3. ¿El desempeño del clasificador se mantiene estable entre periodos de congestión alta y baja dentro del periodo de estudio?

## Hipótesis

- **H1.** La tasa de comisión reciente es el predictor dominante.
- **H2.** Los atributos de actividad de la red aportan una mejora marginal sobre H1, menor que la contribución de la propia tasa.
- **H3.** El desempeño se degrada en los episodios de mayor variación del nivel de comisiones, que son aquellos en los que una estimación correcta resulta más valiosa.

## Conjunto de datos

Fuente: Jesús Graterol, [Bitcoin Blockchain Historical Data](https://www.kaggle.com/datasets/jesusgraterol/bitcoin-blockchain-dataset), Kaggle.

El archivo de origen contiene 810 909 bloques y 13 columnas, de enero de 2009 a octubre de 2023. El estudio se restringe a los bloques posteriores al 1 de enero de 2016, por las razones documentadas abajo: quedan **419 728 bloques**, el 52 % del archivo. Las revisiones de calidad se realizan sobre el archivo completo, antes de aplicar la delimitación.

La variable de interés es `median_fee_rate`: la mediana de las tasas de comisión pagadas por las transacciones incluidas en cada bloque, en satoshis por byte. Se interpreta como el precio aproximado que era necesario ofrecer para ser incluido en ese bloque. El objetivo se construye desplazando esa columna un bloque hacia adelante.

**El CSV no está incluido en este paquete por su tamaño (70 MB).** Descargarlo de Kaggle y colocarlo en `eda/dataset.csv` antes de ejecutar el notebook. El recorte se aplica dentro del notebook; el archivo no se modifica.

## Delimitación del periodo de estudio

La comisión de una transacción se determina por competencia solo cuando el espacio en el bloque es escaso. Si los bloques no se llenan, cualquier transacción entra y la comisión la fijan las reglas del software cliente, no la demanda.

Existieron comisiones en Bitcoin desde 2009: el protocolo las contempla desde el diseño original. Lo que el análisis verifica es otra cuestión: desde cuándo el espacio en bloque constituye un recurso escaso.

| Año | % bloques >900 KB | % tasa cero | Mediana de tasa | Correlación con el siguiente |
|---|---|---|---|---|
| 2012 | 0,0 % | 37,2 % | 59 | 0,049 |
| 2013 | 0,04 % | 7,1 % | 76 | 0,020 |
| 2014 | 1,1 % | 4,5 % | 20 | 0,090 |
| 2015 | 16,8 % | 6,4 % | 19 | 0,124 |
| **2016** | **62,8 %** | 2,1 % | 39 | 0,205 |
| 2017 | 89,3 % | 1,0 % | 134 | 0,702 |
| 2020 | 87,5 % | 0,5 % | 21 | 0,719 |
| 2023 | 96,2 % | 0,3 % | 13 | 0,911 |

Ambas series describen el mismo hecho: la escasez de espacio se establece entre 2015 y 2016. Se adopta el 1 de enero de 2016 como inicio del estudio, conservando 2016 y 2017 porque aportan variedad de condiciones sin incorporar el periodo en que el mecanismo era distinto.

La delimitación se decide a partir de la ocupación de los bloques y de la persistencia de la tasa, **no del desempeño de ningún modelo ni de las métricas de validación o prueba**. No constituye selección de datos guiada por el resultado.

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

`eda/01_eda.ipynb` desarrolla, en veintiuna secciones: formulación del problema y del aprendizaje (1-7); descripción del conjunto de datos y de sus variables (8); calidad de los datos, incluida la coherencia interna de las tasas (9); delimitación del periodo de estudio (10); construcción del objetivo y partición temporal (11); estadísticas descriptivas, distribuciones, tasas cero y valores extremos (12-15); definición y validación de la variable objetivo (16); relaciones entre variables (17); evolución anual (18); prevención de fuga de datos (19); siguientes pasos (20) y limitaciones (21).

## Hallazgos principales

**1. La delimitación del periodo no resuelve el desbalance de clases.**

| Partición | Bloques | Periodo | Mediana de tasa | % clase alta (umbral fijo = 79) |
|---|---|---|---|---|
| Entrenamiento | 293 808 | 2016-01-01 a 2021-05-26 | 29 | 24,92 % |
| Validación | 62 959 | 2021-05-26 a 2022-08-04 | 5 | **0,32 %** |
| Prueba | 62 960 | 2022-08-04 a 2023-10-06 | 11 | **2,44 %** |

Con 0,32 % de positivos en validación, un clasificador que prediga siempre «no alta» obtiene 99,7 % de exactitud sin capacidad predictiva. El nivel de las comisiones no es estacionario ni siquiera dentro de un periodo con mecanismo homogéneo.

**2. Un umbral móvil sí lo resuelve.**

| Partición | Bloques | Periodo | % clase alta (umbral móvil, N = 1008) |
|---|---|---|---|
| Entrenamiento | 293 104 | 2016-01-07 a 2021-05-28 | 24,65 % |
| Validación | 62 808 | 2021-05-28 a 2022-08-05 | 21,47 % |
| Prueba | 62 808 | 2022-08-05 a 2023-10-06 | 24,93 % |

Sensibilidad a la ventana: 24,2 / 19,4 / 24,2 con N = 144 (un día); 24,7 / 21,5 / 24,9 con N = 1008 (una semana); 26,3 / 22,6 / 26,0 con N = 4320 (un mes). El resultado no depende de la elección concreta de N.

La delimitación del periodo y la definición de la clase resuelven problemas distintos: la primera asegura que el fenómeno modelado esté presente en todo el periodo, la segunda se adapta a un nivel de precio que cambia.

**3. La relación con el objetivo es estable dentro del periodo.**

| Época | Pearson | Spearman | Pearson sobre log |
|---|---|---|---|
| 2016-2017 | 0,760 | 0,666 | 0,512 |
| 2018-2019 | 0,851 | 0,565 | 0,645 |
| 2020-2021 | 0,767 | 0,712 | 0,710 |
| 2022-2023 | 0,904 | 0,640 | 0,665 |

Sin tendencia definida y siempre por encima de 0,75 en Pearson, lo que confirma que los bloques posteriores a 2016 describen un único mecanismo de formación del precio.

**4. El tamaño del bloque no informa dentro de este periodo.**

`size` correlaciona 0,067 con el objetivo (Pearson) y 0,002 (Spearman); `tx_count`, 0,109 y 0,113. Una vez que los bloques se llenan de forma habitual, su tamaño no discrimina: casi todos están igualmente llenos y lo que varía es el precio de entrar. Esto acota la hipótesis H2: la mejora marginal esperada procede de las comisiones totales y de variables derivadas aún por construir.

## Decisiones metodológicas

- Periodo de estudio desde el 1 de enero de 2016, justificado en la sección 10 del notebook.
- Partición temporal 70/15/15, sin mezclar el orden de los bloques.
- El umbral se estima solo con entrenamiento.
- Las revisiones de calidad usan el archivo completo, antes del recorte.
- Los diagnósticos de balance usan validación y prueba: son revisiones estructurales del diseño experimental, no estiman parámetros ni seleccionan configuración. Se declara en la sección 19.
- La ocupación del bloque, en la etapa 2, se normalizará por el máximo de una ventana móvil y no por el límite nominal de 1 MB, ya que SegWit (agosto de 2017) cambió la forma de medir la capacidad dentro del periodo de estudio.
- No se entrenan modelos.

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

**Pendiente:** actualizar el documento con la formulación E/T/P, la delimitación del periodo y la definición móvil de la clase, y migrarlo al template IEEE exigido por el curso.

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
