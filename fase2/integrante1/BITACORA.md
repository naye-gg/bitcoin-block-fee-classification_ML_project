# Bitácora del proyecto

Registro de las decisiones tomadas durante el desarrollo del Proyecto Final de Machine Learning
(CS3061, UTEC, 2026-2) y de la evidencia que las sustenta. Sirve para recuperar el porqué de cada
elección al redactar el paper o al responder preguntas en la exposición.

---

## 1. Elección del tema

### Dominio: criptomonedas y blockchain

Se evaluaron varias líneas antes de decidir. La trampa principal del dominio es la **predicción de
precio**: el baseline honesto (el precio de ayer) es casi imbatible, la mayoría de tutoriales
públicos incurren en fuga de datos al usar particiones aleatorias sobre series temporales, y con
modelos lineales los resultados son pobres.

Opciones consideradas, con sus datasets:

| Tema | Dataset | Observación |
|---|---|---|
| Detección de cuentas fraudulentas en Ethereum | `vagifa/ethereum-frauddetection-dataset` (9 841 × 51) | Features interpretables, pero muy usado en proyectos de curso |
| Transacciones ilícitas en Bitcoin | Elliptic (203 769 nodos, 166 features) | Benchmark académico, pero features anonimizadas |
| Direcciones de ransomware | BitcoinHeist, UCI id 526 (2,9 M × 10) | Desbalance extremo (<1 %), 225 MB |
| Régimen de volatilidad | OHLCV de 50 criptomonedas | Viable, pero temáticamente más común |
| **Comisiones y congestión de Bitcoin** | `jesusgraterol/bitcoin-blockchain-dataset` (810 909 × 13) | **Elegido** |
| Contratos Ponzi en Ethereum | `polarwolf/ponzi-scheme-contracts-on-ethereum` | Original, pero solo 318 positivos |

### Por qué comisiones de Bitcoin

- Problema con usuario real y perdedor identificable: quien paga de más tira dinero, quien paga de
  menos espera horas.
- Variables con significado físico y documentación pública, no anonimizadas.
- No es predicción de precio, por lo que evita la crítica asociada.
- Volumen suficiente: 810 909 bloques.

---

## 2. El dataset

**Fuente:** Jesús Graterol, *Bitcoin Blockchain Historical Data*, Kaggle. Generado a partir de la
API pública de Mempool.space.

810 909 filas × 13 columnas, de enero de 2009 a octubre de 2023. Cada fila es un **bloque**, no una
transacción.

Columnas: `height`, `timestamp`, `size`, `tx_count`, `difficulty`, `median_fee_rate`,
`avg_fee_rate`, `total_fees`, `fee_range_min`, `fee_range_max`, `input_count`, `output_count`,
`output_amount`. Las tasas se expresan en **satoshis por byte**.

### Implicación de la agregación por bloque

Al no disponer de transacciones individuales, el proyecto **no modela la decisión de un usuario**
sino el nivel agregado del mercado. Esta distinción obligó a retirar del planteamiento inicial la
afirmación sobre prioridad de transacciones, conservándola solo como motivación.

Consecuencias que se declaran como limitaciones:

- **Sesgo de selección:** solo se observan transacciones confirmadas. Las ofertas insuficientes no
  quedan registradas; se observa el resultado de la subasta, no el conjunto de ofertas.
- **Ausencia de mempool:** la cola de transacciones pendientes es un estado transitorio que no
  puede reconstruirse retrospectivamente. Por eso los trabajos que la usan recolectaron datos en
  tiempo real.
- **Falacia ecológica:** una relación observada a nivel agregado no se transfiere automáticamente
  al nivel individual.

---

## 3. Formulación del problema

**Paradigma:** aprendizaje supervisado, clasificación binaria, sobre datos secuenciales, con clases
moderadamente desbalanceadas.

No es regresión (la salida es una categoría), no es multiclase (dos clases), no es no supervisado
(hay etiquetas), y no es *forecasting* en sentido estricto (no se modela la serie completa a futuro
sino el siguiente punto, con horizonte de un paso).

| | |
|---|---|
| **Tarea (T)** | Predecir si la tasa mediana del bloque *t+1* superará el percentil 75 de los bloques recientes |
| **Experiencia (E)** | 419 727 pares (bloque, bloque siguiente), 13 atributos por bloque |
| **Desempeño (P)** | F1 de la clase positiva; secundarias: precisión, exhaustividad, AP/PR-AUC, ROC-AUC |

La exactitud no se usa como métrica principal: con tasa base cercana al 25 %, predecir siempre la
clase mayoritaria da ~75 % de exactitud y F1 nulo sobre la clase de interés.

---

## 4. El hallazgo del EDA: el umbral fijo no es evaluable

### Qué se encontró

El planteamiento inicial definía «comisión alta» con un umbral fijo: el percentil 75 del conjunto
de entrenamiento. Es la decisión estándar, y se calculó solo con entrenamiento, que es lo correcto.

Al revisar el balance en las demás particiones apareció el problema:

| Partición | % clase alta (umbral fijo = 79) |
|---|---:|
| Entrenamiento | 24,92 % |
| Validación | **0,32 %** |
| Prueba | 2,44 % |

Con 0,32 % de positivos, un clasificador que prediga siempre «no alta» obtiene 99,7 % de exactitud
sin haber aprendido nada. Ninguna de las dos particiones permitía evaluar el modelo.

### Por qué ocurre

El nivel de comisiones no es estacionario. La mediana trimestral pasa de 180 en 2017Q2 a 4 en
2021Q3. Un umbral constante no describe la misma noción de «tasa alta» a lo largo del periodo:
separa épocas, no bloques caros de bloques baratos.

### Lo que no funcionó

Se probaron nueve fechas de corte distintas manteniendo el umbral fijo. En todas el desbalance
persistía, y en varias la validación quedaba con cero o casi cero positivos. **Recortar el periodo
no resuelve el problema**: la no estacionariedad existe a cualquier escala.

---

## 5. La solución: umbral móvil

### Definición

```
θ_t = P75( tasa_{t-1007}, ..., tasa_t )
y_t = 1  si  tasa_{t+1} > θ_t ;  0 en caso contrario
```

La ventana de 1 008 bloques equivale a una semana aproximadamente (un bloque cada ~10 minutos).

El umbral se recalcula en cada bloque: en 2017 valdría alrededor de 150, en 2022 alrededor de 9.
Pero la etiqueta siempre significa lo mismo: **más caro de lo normal para este momento**.

### Resultado

| Partición | Umbral fijo | Umbral móvil |
|---|---:|---:|
| Entrenamiento | 24,92 % | 24,65 % |
| Validación | 0,32 % | 21,47 % |
| Prueba | 2,44 % | 24,93 % |

### Sensibilidad a la ventana

| N | Equivale a | Train | Val | Test |
|---|---|---:|---:|---:|
| 144 | 1 día | 24,2 % | 19,4 % | 24,2 % |
| 1 008 | 1 semana | 24,7 % | 21,5 % | 24,9 % |
| 4 320 | 1 mes | 26,3 % | 22,6 % | 26,0 % |

El balance es estable en las tres, de modo que el resultado no depende de la elección concreta de
N. Se conservó 1 008 porque una semana comprende un ciclo completo de actividad de la red.

### Ausencia de fuga

La ventana usa bloques **hasta el actual inclusive**; el objetivo corresponde al bloque
**siguiente**. Por construcción no hay información del futuro.

### Cambio en el significado de la tarea

| | Umbral fijo | Umbral móvil |
|---|---|---|
| La clase significa | tasa alta en términos históricos | tasa alta respecto al nivel reciente |
| Responde | ¿va a estar cara la red? | ¿va a subir respecto a como ha estado? |
| Utilidad | comparar épocas | decidir si emitir ahora o esperar |

---

## 6. Delimitación del periodo: desde 2016

### Motivación

La comisión se determina por competencia solo cuando el espacio en bloque es escaso. Si los bloques
no se llenan, entra cualquier transacción y la comisión la fijan las reglas del software cliente.

| Año | % bloques >900 KB | % tasa cero | Mediana | Corr. con el siguiente |
|---|---:|---:|---:|---:|
| 2012 | 0,0 % | 37,2 % | 59 | 0,049 |
| 2013 | 0,04 % | 7,1 % | 76 | 0,020 |
| 2014 | 1,1 % | 4,5 % | 20 | 0,090 |
| 2015 | 16,8 % | 6,4 % | 19 | 0,124 |
| **2016** | **62,8 %** | 2,1 % | 39 | 0,205 |
| 2017 | 89,3 % | 1,0 % | 134 | 0,702 |
| 2020 | 87,5 % | 0,5 % | 21 | 0,719 |
| 2023 | 96,2 % | 0,3 % | 13 | 0,911 |

**No es que antes no hubiera comisiones** —existen desde 2009 y el protocolo las contempla desde el
diseño original— sino que no se determinaban por competencia entre usuarios.

### Precisión importante

El recorte y el umbral móvil **resuelven problemas distintos**:

- El recorte homogeneiza el **mecanismo** de formación del precio.
- El umbral móvil se adapta al **nivel**, que sigue siendo no estacionario dentro del periodo
  recortado (mediana anual: 134 en 2017, 8 en 2018).

La prueba de que son independientes: con el recorte aplicado y umbral fijo, la validación seguía en
0,32 %.

### Criterio de decisión

Se adoptó el 1 de enero de 2016 por ser el primer año en que la mayoría de bloques opera cerca del
límite. La delimitación se decidió a partir de la ocupación y la persistencia de la tasa, **no del
desempeño de ningún modelo**, por lo que no constituye selección guiada por el resultado.

Quedan 419 727 pares; 418 720 etiquetables tras descartar las 1 007 primeras filas, donde la
ventana del umbral no está completa.

---

## 7. Anomalías de calidad detectadas

| Anomalía | Conteo (2016-2023) | Tratamiento |
|---|---:|---|
| Tasa mediana igual a cero | 3 299 | Razones como diferencias de `log1p`, no cocientes |
| Intervalos negativos entre bloques | 1 721 (mín. −1 911 s) | Truncados en cero; es regla del protocolo, no error |
| Mediana fuera de `[fee_range_min, fee_range_max]` | 2 512 | Documentado; afecta solo a `amplitud_rango` |
| Bloques con una sola transacción (coinbase) | 2 915 | Conservados |
| Saltos en `height` | 0 | — |
| Valores faltantes y duplicados | 0 | — |

Los intervalos negativos no son un error de los datos: el protocolo admite que la marca de tiempo
de un bloque sea anterior a la del precedente mientras supere la mediana de los once anteriores.

### Correcciones de análisis durante el EDA

- **La correlación de 0,05 era un artefacto.** Sobre el archivo completo, Pearson entre tasa actual
  y siguiente da 0,05. Separando por épocas: 0,042 en 2009-2011 frente a **0,819 en 2017-2019** y
  0,817 en 2020-2023. El valor agregado resultaba de mezclar un periodo sin competencia con otro
  donde sí la hay, más la influencia de valores extremos.
- **El tamaño del bloque cambia de lectura según el periodo.** Sobre el archivo completo,
  `size` correlaciona 0,504 con el objetivo bajo Spearman; dentro del periodo recortado, 0,002. La
  asociación aparente provenía de que ambas variables separan épocas. Dentro del régimen actual,
  casi todos los bloques están igual de llenos y lo que varía es el precio de entrar.

---

## 8. Etapa 1 (P1)

Se produjeron dos versiones completas y autónomas del EDA, una con el periodo completo y otra
restringida desde 2016. **Se entregó la versión recortada**, que obtuvo 20/20.

Contenido de la entrega: formulación del problema con E/T/P, alcance, preguntas de investigación,
hipótesis, plan de trabajo, descripción del dataset, calidad de los datos, delimitación del
periodo, construcción del objetivo y partición, EDA, prevención de fuga de datos, siguientes pasos
y limitaciones.

---

## 9. Etapa 2 (P2) — parte correspondiente a baselines

### Ingeniería de características

22 variables, todas con información hasta el bloque actual:

| Grupo | Variables |
|---|---|
| Nivel relativo | `r_tasa`, rezagos 1/2/3/6, medias móviles 6 y 144, volatilidad 144, `tendencia` |
| Dispersión | `disp_avg_med`, `amplitud_rango` |
| Actividad | `ocupacion`, `tx_relativo`, `dt_bloque`, `dt_media6`, `tam_medio_tx`, `ent_por_tx`, `sal_por_tx` |
| Calendario | `hora_sin`, `hora_cos`, `dia_sin`, `dia_cos` |

Decisiones justificadas:

- **Razones como diferencias de logaritmos**, porque el nivel absoluto no es estacionario y
  `log1p` está definida en cero.
- **Ocupación normalizada por ventana móvil** y no por el límite nominal de 1 MB, porque SegWit
  (agosto de 2017) cambió la forma de medir la capacidad dentro del periodo de estudio.
- **Imputación y escalamiento dentro del pipeline**, ajustados solo con entrenamiento.

### Hallazgos del EDA orientado a la predicción

**Persistencia.** Cuando el bloque actual supera su umbral local, el siguiente lo supera el 52,8 %
de las veces; cuando no lo supera, solo el 15,5 %. Esta asimetría sustenta H1 y justifica el
baseline.

**Estacionalidad semanal.** La clase positiva va de 11,3 % los domingos a 32,4 % los jueves, y de
15,9 % a 37,7 % según la hora UTC. La explicación está en la interacción entre el ciclo semanal de
la red y la ventana del umbral, que promedia una semana completa: un bloque de domingo se compara
contra un nivel que incluye la actividad de entre semana.

Esto conlleva una advertencia metodológica registrada en limitaciones: **parte de la señal de
calendario procede de la propia definición de la clase**, no del mercado de comisiones.

### Resultados

| Modelo | Partición | F1 | Precisión | Exhaustividad | AP | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Persistencia | Validación | 0,5398 | 0,5405 | 0,5391 | — | — |
| Persistencia | Prueba | **0,5763** | 0,5770 | 0,5755 | — | — |
| Reg. logística | Validación | **0,5572** | 0,5157 | 0,6060 | 0,5929 | 0,8090 |
| Reg. logística | Prueba | 0,5704 | 0,5478 | 0,5950 | 0,6166 | 0,8061 |

**La regresión logística gana en validación y pierde en prueba.**

Verificaciones realizadas sobre ese resultado:

- **No se debe a la transferencia del umbral.** El umbral óptimo de prueba sería 0,271 en vez de
  0,312, y la diferencia cuesta 0,007 en F1. Aun con el umbral óptimo de prueba —diagnóstico que no
  interviene en la selección— el modelo alcanzaría 0,5776 frente a 0,5763 de la persistencia.
- **El modelo sí discrimina:** AP de 0,6166 en prueba sobre una tasa base de 0,2493, es decir 2,5
  veces la referencia aleatoria, y ROC-AUC de 0,806.

Interpretación: la señal aprovechable por un modelo lineal coincide en buena parte con la inercia
que la persistencia ya captura.

### El ajuste del umbral pesa más que la regularización

| Configuración | F1 con umbral 0,5 | Umbral ajustado | F1 ajustado |
|---|---:|---:|---:|
| L2, C = 1, sin ponderación | 0,4174 | 0,312 | 0,5576 |
| L2, C = 1, `balanced` | 0,5509 | 0,534 | 0,5573 |

Las dieciséis configuraciones evaluadas caben en un rango de 0,0003 en F1: con 288 784
observaciones frente a 22 variables no hay margen de sobreajuste que la penalización pueda
corregir. La elección entre configuraciones es arbitraria dentro del ruido y así se reporta.

### Coeficientes

Dominan las variables de nivel relativo: media móvil de 144 bloques (0,812), media de 6 (0,487) y
volatilidad reciente (0,431), lo que respalda H1. El signo positivo de la volatilidad es coherente
con que la congestión llega en rachas.

Las variables de actividad pesan poco; `hora_sin` (−0,332) y `dia_sin` (0,171) entran con peso
apreciable. Respecto a H2, la mejora marginal procede del calendario antes que del volumen.

Los coeficientes no admiten lectura causal: las variables de nivel relativo están fuertemente
correlacionadas entre sí.

### Pregunta abierta para el resto del proyecto

Si existe estructura adicional en la relación entre el estado de la red y la congestión del bloque
siguiente, no es de naturaleza lineal. Los modelos de ensamble deberían separarse de la
persistencia donde el modelo lineal no lo consigue. Si tampoco lo logran, la conclusión del
proyecto sería que estas variables agregadas por bloque no contienen información explotable más
allá de la inercia.

---

## 10. Referencias verificadas

- J. Graterol. *Bitcoin Blockchain Historical Data*, Kaggle.
  https://www.kaggle.com/datasets/jesusgraterol/bitcoin-blockchain-dataset
- J. Graterol. *Generador y esquema del dataset*.
  https://github.com/jesusgraterol/bitcoin-blockchain-dataset-builder
- M. Möser y R. Böhme. «Trends, Tips, Tolls: A Longitudinal Study of Bitcoin Transaction Fees».
  *Financial Cryptography and Data Security*, 2nd Workshop on BITCOIN Research, LNCS 8976,
  Springer, 2015, pp. 19-33. DOI: 10.1007/978-3-662-48051-9_2
  → Documenta que los cambios de régimen en las comisiones se explican por cambios en el software
  cliente. Sustenta la delimitación del periodo.
- D. Easley, M. O'Hara y S. Basu. «From Mining to Markets: The Evolution of Bitcoin Transaction
  Fees». *Journal of Financial Economics* 134(1), 2019, pp. 91-109.
  DOI: 10.1016/j.jfineco.2019.03.004
  → Marco económico de por qué existen las comisiones y qué variables las determinan.
- G. Huberman, J. D. Leshno y C. Moallemi. «Monopoly without a Monopolist: An Economic Analysis of
  the Bitcoin Payment System». *The Review of Economic Studies* 88(6), 2021, pp. 3011-3040.
  DOI: 10.1093/restud/rdab014
- E. Tedeschi, T. A. Nordmo, D. Johansen y H. D. Johansen. «On Optimizing Transaction Fees in
  Bitcoin using AI: Investigation on Miners Inclusion Pattern». *ACM Transactions on Internet
  Technology* 22(3), art. 77, 2022. DOI: 10.1145/3528669 — acceso abierto.
  → Antecedente de ML más cercano; usa estados del mempool, que este proyecto no tiene.
- A. K. Singh y A. S. Hafid. «Prediction of Transaction Confirmation Time in Ethereum Blockchain
  Using Machine Learning». *Advances in Intelligent Systems and Computing* 1010, 2020, pp. 126-133.
  Preprint: arXiv:1911.11592
  → Estructura de comparación entre modelo estadístico de referencia y modelos de ML.
- «Bitcoin Transaction Confirmation Time Prediction: A Classification View». Springer, 2022.
  DOI: 10.1007/978-3-031-20891-1_12
  → Respalda formular la tarea como clasificación por intervalos y usar un baseline de una sola
  variable. **Verificar la lista de autores en el DOI antes de citar.**

---

## 11. Decisiones que conviene poder defender

| Decisión | Justificación en una frase |
|---|---|
| Nivel de bloque, no de transacción | El dataset está agregado; no permite afirmar el tiempo de confirmación individual |
| Umbral móvil, no fijo | El nivel de comisiones no es estacionario; un umbral fijo separa épocas |
| N = 1 008 | Una semana comprende un ciclo completo de actividad; el balance es estable para 144, 1 008 y 4 320 |
| Periodo desde 2016 | Primer año en que la mayoría de bloques opera cerca del límite (62,8 % frente a 1,1 % en 2014) |
| Partición cronológica | Una partición aleatoria permitiría entrenar con bloques posteriores a los evaluados |
| F1 y no exactitud | Con tasa base de 25 %, la clase mayoritaria da 75 % de exactitud y F1 nulo |
| Baseline de persistencia | Es la referencia exigente; la clase mayoritaria tiene F1 nulo y no informa |
| Umbral de decisión 0,312 | Con clases desbalanceadas, 0,5 no optimiza F1; ajustado solo sobre validación |
| Mirar el balance de validación y prueba | Revisión estructural del diseño, no estimación de parámetros ni selección por desempeño |
