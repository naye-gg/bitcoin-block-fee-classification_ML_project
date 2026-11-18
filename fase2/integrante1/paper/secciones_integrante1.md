# Secciones del paper — Integrante 1

Texto listo para insertar en el documento IEEE. Cubre: introducción y formulación del problema,
datos y EDA, la parte de preprocesamiento correspondiente a los baselines, y la metodología de los
modelos de referencia.

Las cifras provienen de `notebooks/01_baselines.ipynb` y de `results/baselines_resultados.csv`.
Las figuras citadas están en `results/figures/`.

---

## II. Introducción y formulación del problema

### A. Contexto

Cuando una persona emite una transacción en Bitcoin debe elegir qué comisión ofrece a los mineros.
La decisión tiene costo en ambos sentidos: una comisión insuficiente puede dejar la transacción sin
confirmar durante horas, mientras que una comisión excesiva constituye un sobrecosto irrecuperable,
dado que las transacciones son irreversibles.

El espacio en cada bloque es limitado y los mineros seleccionan qué transacciones incluyen. Cuando
la demanda supera la capacidad disponible se forma una subasta implícita en la que los usuarios
compiten ofreciendo comisiones mayores, y el precio resultante varía de forma considerable en
cuestión de horas. Las carteras incorporan estimadores de comisión, pero su desempeño es limitado
precisamente en los episodios de congestión, que son aquellos en los que una estimación correcta
resulta más valiosa.

### B. Objetivo

El objetivo del proyecto es determinar si, a partir del estado observable de la red hasta el bloque
actual, es posible anticipar si la comisión del siguiente bloque será alta respecto al nivel vigente
del mercado.

El resultado no constituye un estimador de comisión completo, sino su primer componente: la
estimación del nivel de precio del mercado, sobre la cual un estimador operativo aplicaría después
el criterio de urgencia de cada usuario.

### C. Formulación como problema de aprendizaje

El problema se formula como una tarea de **aprendizaje supervisado de clasificación binaria** sobre
datos secuenciales, con clases moderadamente desbalanceadas.

**Tarea (T).** Dado el bloque *t* y su historia, predecir si la tasa mediana de comisión del bloque
*t+1* superará el percentil 75 de los bloques recientes:

```
θ_t = P75( tasa_{t-1007}, ..., tasa_t )
y_t = 1  si  tasa_{t+1} > θ_t ;  0 en caso contrario
```

La ventana de 1 008 bloques equivale aproximadamente a una semana, dado que el protocolo ajusta la
dificultad para que se mine un bloque cada diez minutos en promedio.

**Experiencia (E).** 419 727 pares (bloque, bloque siguiente) construidos a partir del registro de la
cadena de bloques entre enero de 2016 y octubre de 2023, con trece atributos observables por bloque.

**Desempeño (P).** F1 sobre la clase positiva como métrica principal, acompañada de precisión,
exhaustividad, precisión promedio (AP) y ROC-AUC. No se emplea la exactitud: con una tasa base
cercana al 25 %, un clasificador que prediga siempre la clase mayoritaria alcanza una exactitud
próxima al 75 % con un F1 nulo sobre la clase de interés. Toda comparación se realiza contra una
referencia de persistencia, descrita en la Sección V.

### D. Alcance

Tres delimitaciones se derivan de la naturaleza de los datos y condicionan la interpretación de
cualquier resultado:

1. **Se predice el nivel de comisión, no la ocupación del bloque.** La ocupación es la condición que
   da origen al mercado de comisiones, pero dentro del periodo de estudio los bloques operan de
   forma habitual cerca de su capacidad, por lo que su llenado no es un evento informativo.
2. **Se modela el mercado agregado, no la decisión individual.** La unidad de observación es el
   bloque, que resume cientos o miles de transacciones. El conjunto de datos no contiene el
   historial de transacciones individuales.
3. **Solo se observan transacciones confirmadas.** Las ofertas insuficientes no quedan registradas,
   de modo que se observa el resultado de la subasta y no el conjunto de ofertas.

### E. Preguntas de investigación e hipótesis

1. ¿La tasa de comisión del bloque actual permite anticipar la del siguiente por encima de lo que
   logra una regla de persistencia?
2. ¿Qué atributos del estado de la red aportan información adicional a la tasa reciente?
3. ¿El desempeño se mantiene estable entre periodos de congestión alta y baja?

**H1.** La tasa de comisión reciente es el predictor dominante.
**H2.** Los atributos de actividad de la red aportan una mejora marginal sobre H1, menor que la
contribución de la propia tasa.
**H3.** El desempeño se degrada en los episodios de mayor variación del nivel de comisiones.

---

## III. Datos y análisis exploratorio

### A. Origen y características

Se emplea el conjunto *Bitcoin Blockchain Historical Data*, publicado por J. Graterol en Kaggle y
construido a partir de la API pública de Mempool.space. El archivo contiene 810 909 bloques y trece
variables agregadas por bloque, desde enero de 2009 hasta octubre de 2023. Cada fila representa un
bloque individual.

La variable de interés es `median_fee_rate`: la mediana de las tasas de comisión pagadas por las
transacciones incluidas en el bloque, expresada en satoshis por byte. Se interpreta como el precio
aproximado que era necesario ofrecer para ser incluido en ese bloque.

Siguiendo la delimitación establecida en la primera entrega, el estudio se restringe a los bloques
posteriores al 1 de enero de 2016. Antes de esa fecha los bloques no alcanzaban su límite de
capacidad de forma habitual —el 1,1 % en 2014 frente al 62,8 % en 2016— y en ausencia de escasez la
comisión no se determinaba por competencia entre usuarios, sino por reglas fijas del software
cliente. El periodo conserva 419 727 pares bloque-siguiente. Al descartar las 1 007 primeras
observaciones, en las que la ventana del umbral todavía no está completa, quedan 418 720
observaciones etiquetables.

### B. Partición experimental

La partición es cronológica, en proporción 70 / 15 / 15, sin alterar el orden de los bloques. Una
partición aleatoria permitiría entrenar con bloques posteriores a los evaluados.

TABLA I. PARTICIÓN EXPERIMENTAL

| Partición | Bloques | Periodo | Clase positiva |
|---|---:|---|---:|
| Entrenamiento | 293 104 | 2016-01-07 a 2021-05-28 | 24,65 % |
| Validación | 62 808 | 2021-05-28 a 2022-08-05 | 21,47 % |
| Prueba | 62 808 | 2022-08-05 a 2023-10-06 | 24,93 % |

La tasa de positivos de validación es sensiblemente menor que la de prueba. Esta diferencia es
relevante al transferir el umbral de decisión y se retoma en la Sección VII.

### C. Calidad de los datos

La revisión del periodo de estudio identifica cuatro condiciones con consecuencias directas sobre el
diseño de las variables:

1. **No existen saltos en `height`.** La cadena es continua, de modo que los rezagos y las ventanas
   móviles por posición equivalen a rezagos por bloque.
2. **3 299 bloques presentan tasa mediana igual a cero.** Cualquier cociente que use esa columna
   como denominador produciría valores no definidos; las razones entre tasas se construyen como
   diferencias de logaritmos mediante `log1p`.
3. **1 721 intervalos entre bloques son negativos**, con un mínimo de −1 911 segundos. No se trata
   de un error de los datos: el protocolo admite que la marca de tiempo de un bloque sea anterior a
   la del precedente mientras supere la mediana de los once anteriores. Al carecer de interpretación
   física, los intervalos negativos se truncan en cero.
4. **2 512 bloques presentan la mediana fuera del rango publicado** `[fee_range_min,
   fee_range_max]`, situación lógicamente imposible si las tres columnas describieran el mismo
   conjunto de transacciones. La inconsistencia procede de la fuente. Esas columnas se utilizan
   únicamente para construir la amplitud del rango, de modo que el efecto queda acotado a una
   variable.

No se registran valores faltantes ni duplicados.

### D. Hallazgos exploratorios relevantes para la predicción

**1) Persistencia del nivel relativo.** Se define la posición de la tasa actual respecto a su
referencia local como la diferencia entre sus logaritmos. La distribución de esta magnitud difiere
de forma apreciable entre clases (Fig. 1): los bloques cuyo sucesor resulta de clase positiva
tienden a presentar ya una tasa por encima de su referencia.

La tabla de transición cuantifica el efecto. Cuando el bloque actual supera su umbral local, el
siguiente lo supera el **52,8 %** de las veces; cuando no lo supera, solo el **15,5 %**. Esta
asimetría sustenta la hipótesis H1 y justifica la elección del baseline de persistencia.

**2) Estacionalidad semanal.** La proporción de clase positiva varía entre **11,3 % los domingos y
32,4 % los jueves**, y entre 15,9 % y 37,7 % según la hora UTC (Fig. 2). La tasa mediana acompaña el
patrón: 18 los domingos frente a 37 los jueves.

El efecto admite una explicación en términos de la definición de la clase. El umbral se calcula
sobre 1 008 bloques, aproximadamente una semana completa, de modo que promedia días laborables y
fin de semana. Un bloque de domingo se compara contra un nivel que incorpora la actividad de entre
semana y rara vez lo supera; uno de jueves lo supera con frecuencia. En consecuencia, las variables
de calendario aportan señal propia, pero parte de esa señal procede de la interacción entre el ciclo
semanal de la red y la ventana empleada para definir el umbral, y no de una propiedad del mercado de
comisiones. Este punto se recoge en las limitaciones.

---

## IV. Preprocesamiento e ingeniería de características

Se construyen veintidós variables a partir de información disponible **hasta el bloque actual**.
Ninguna utiliza la tasa del bloque siguiente. La ausencia de fuga se verifica de forma explícita
perturbando la variable objetivo y comprobando que ninguna columna de entrada se modifica.

TABLA II. GRUPOS DE VARIABLES

| Grupo | Variables | Fenómeno |
|---|---|---|
| Nivel relativo | `r_tasa`, rezagos 1, 2, 3 y 6, medias móviles de 6 y 144, volatilidad de 144, `tendencia` | Posición de la tasa respecto a su referencia y dirección del movimiento |
| Dispersión | `disp_avg_med`, `amplitud_rango` | Diferencia entre lo que pagan las transacciones de un mismo bloque |
| Actividad | `ocupacion`, `tx_relativo`, `dt_bloque`, `dt_media6`, `tam_medio_tx`, `ent_por_tx`, `sal_por_tx` | Congestión y demanda acumulada |
| Calendario | `hora_sin`, `hora_cos`, `dia_sin`, `dia_cos` | Estacionalidad diaria y semanal |

Tres decisiones requieren justificación:

**Razones expresadas como diferencias de logaritmos.** El nivel absoluto de las comisiones no es
estacionario dentro del periodo de estudio: la mediana anual pasa de 134 en 2017 a 8 en 2018. Usar
`median_fee_rate` en bruto obligaría al modelo a aprender umbrales que dejan de ser válidos al
cambiar de periodo. La diferencia logarítmica respecto al umbral local es comparable entre épocas y
está definida cuando la tasa es cero.

**Ocupación normalizada por ventana móvil.** La activación de SegWit en agosto de 2017 modificó la
forma de medir la capacidad del bloque dentro del periodo de estudio. Normalizar por el límite
nominal de 1 MB produciría ocupaciones superiores a la unidad en la parte reciente de la serie, por
lo que se emplea el máximo observado en una ventana móvil de 4 320 bloques.

**Imputación y escalamiento dentro del pipeline.** Ambas transformaciones se ajustan únicamente con
entrenamiento, de modo que ningún estadístico de validación o prueba influye en la transformación.
Las 4 319 primeras observaciones de entrenamiento se descartan por corresponder al calentamiento de
la ventana móvil más larga; validación y prueba no presentan valores faltantes. Se añade una purga
de una observación entre particiones, correspondiente a la última fila de entrenamiento, cuya
etiqueta se construye mirando el primer bloque de la partición siguiente.

---

## V. Metodología: modelos de referencia

### A. Persistencia histórica

La referencia más exigente para este problema no es la clase mayoritaria sino la regla que predice,
para el bloque siguiente, la clase observada en el actual:

```
ŷ_t = 1[ tasa_t > θ_t ]
```

No requiere entrenamiento y utiliza únicamente información disponible en el momento de predecir. Su
fundamento empírico es la tabla de transición de la Sección III-D. Cualquier modelo que no la supere
no estaría aportando sobre la inercia del mercado.

### B. Regresión logística regularizada

Se emplea regresión logística dentro de un pipeline con imputación por mediana y estandarización.
La búsqueda abarca dos penalizaciones (L1 y L2), cuatro valores del parámetro de regularización
*C* ∈ {0,01; 0,1; 1; 10} y dos tratamientos del desbalance (`class_weight` nulo o balanceado).

Para cada configuración se ajusta además el **umbral de decisión** maximizando F1 sobre validación,
en lugar de conservar el valor por defecto de 0,5. Con clases desbalanceadas, el corte por defecto
no corresponde al punto que optimiza la métrica declarada.

La selección se realiza exclusivamente sobre validación; la partición de prueba se evalúa una sola
vez, una vez fijadas todas las decisiones de modelado.

TABLA III. EFECTO DEL AJUSTE DEL UMBRAL (VALIDACIÓN)

| Configuración | F1 con umbral 0,5 | Umbral ajustado | F1 ajustado |
|---|---:|---:|---:|
| L2, C = 1, sin ponderación | 0,4174 | 0,312 | 0,5576 |
| L2, C = 1, `balanced` | 0,5509 | 0,534 | 0,5573 |

Dos observaciones se desprenden de la búsqueda completa. En primer lugar, el ajuste del umbral
influye mucho más que la regularización: con el corte por defecto el F1 es 0,417, y con el umbral
ajustado asciende a 0,558. La opción `balanced` alcanza un resultado equivalente por otra vía, ya
que reponderar las clases desplaza implícitamente el punto de operación. En segundo lugar, las
dieciséis configuraciones evaluadas quedan dentro de un rango de 0,0003 en F1, lo que resulta
coherente con 288 784 observaciones frente a veintidós variables: no existe margen de sobreajuste
que la penalización pueda corregir. La configuración seleccionada —L2, *C* = 1, sin ponderación,
umbral 0,312— es la de mayor F1 ajustado, si bien la elección entre configuraciones es arbitraria
dentro del ruido.

---

## VI. Resultados de los modelos de referencia

TABLA IV. DESEMPEÑO DE LOS MODELOS DE REFERENCIA

| Modelo | Partición | F1 | Precisión | Exhaustividad | AP | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Persistencia | Validación | 0,5398 | 0,5405 | 0,5391 | — | — |
| Persistencia | Prueba | **0,5763** | 0,5770 | 0,5755 | — | — |
| Reg. logística | Validación | **0,5572** | 0,5157 | 0,6060 | 0,5929 | 0,8090 |
| Reg. logística | Prueba | 0,5704 | 0,5478 | 0,5950 | 0,6166 | 0,8061 |

**La regresión logística supera a la persistencia en validación pero no en prueba.** La diferencia a
favor del modelo lineal en validación es de 0,0174 en F1; en prueba, la persistencia aventaja al
modelo en 0,0059.

Conviene precisar el alcance de este resultado. El modelo **sí discrimina**: la precisión promedio
sobre prueba es 0,6166 frente a una tasa base de 0,2493, es decir 2,5 veces la referencia aleatoria,
y el ROC-AUC alcanza 0,806. Lo que no consigue es superar a la persistencia en un punto de operación
concreto.

El resultado **no se explica por la transferencia del umbral**. El umbral óptimo sobre prueba sería
0,271 en lugar de 0,312, y la diferencia cuesta 0,007 en F1. Incluso ajustando el umbral sobre la
propia partición de prueba —procedimiento que invalidaría la evaluación y que se calcula únicamente
como diagnóstico— el modelo alcanzaría 0,5776 frente a 0,5763 de la persistencia, una diferencia no
atribuible al modelo.

La interpretación razonable es que la señal aprovechable por un modelo lineal sobre estas variables
coincide en buena parte con la inercia que la persistencia ya captura. Si existe estructura adicional
en la relación entre el estado de la red y la congestión del bloque siguiente, no es de naturaleza
lineal.

### Interpretación de los coeficientes

Con las variables estandarizadas, la magnitud de los coeficientes es comparable entre ellas. Los tres
mayores corresponden a variables de nivel relativo: la media móvil de 144 bloques (0,812), la de 6
bloques (0,487) y la volatilidad reciente (0,431). Esto respalda la hipótesis H1.

El signo positivo de la volatilidad es interpretable: cuando la tasa ha oscilado de forma marcada en
las últimas observaciones aumenta la probabilidad de superar el umbral local en el bloque siguiente,
lo que es coherente con la idea de que la congestión llega en rachas.

Las variables de actividad —ocupación, tamaño medio de transacción, entradas y salidas por
transacción— presentan coeficientes pequeños, en línea con lo documentado en la primera entrega:
dentro del periodo de estudio los bloques operan habitualmente cerca de su capacidad, de modo que su
tamaño apenas discrimina. En cambio `hora_sin` (−0,332) y `dia_sin` (0,171) entran con peso
apreciable, en coherencia con la estacionalidad descrita en la Sección III-D. Esta lectura, respecto
a H2, sugiere que la mejora marginal procede del calendario antes que de las variables de volumen.

El reparto del peso entre variables de nivel relativo no es estable, ya que están fuertemente
correlacionadas entre sí, por lo que los coeficientes no deben interpretarse como una medida de
importancia causal.

---

## Limitaciones correspondientes a esta parte

- Las variables se construyen sobre datos agregados por bloque. No se dispone del estado del
  mempool, información que la literatura sobre estimación de comisiones considera central y que no
  puede reconstruirse retrospectivamente por tratarse de un estado transitorio.
- El umbral de decisión se ajusta sobre una partición de validación cuya tasa de positivos (21,47 %)
  difiere de la de prueba (24,93 %), lo que penaliza ligeramente la transferencia.
- Parte de la señal de calendario procede de la propia definición de la clase, al calcularse el
  umbral sobre una ventana de aproximadamente una semana.
- Los coeficientes del modelo lineal no admiten lectura causal por la correlación entre variables de
  nivel relativo.
- Las 2 512 inconsistencias de la fuente afectan a `fee_range_min` y `fee_range_max`, y por tanto a
  la variable `amplitud_rango`.

---

## Figuras disponibles

| Archivo | Contenido | Sección sugerida |
|---|---|---|
| `eda_posicion_relativa.png` | Distribución de la posición relativa por clase y tabla de transición | III-D |
| `eda_calendario.png` | Tasa de clase positiva por hora y por día de la semana | III-D |
| `ajuste_umbral_validacion.png` | F1 frente al umbral de decisión sobre validación | V-B |
| `pr_y_matriz_confusion.png` | Curva precisión-exhaustividad y matriz de confusión de prueba | VI |
| `coeficientes_logistica.png` | Coeficientes de la regresión logística | VI |
