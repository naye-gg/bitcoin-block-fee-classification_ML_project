# Verificación de la entrega — Integrante 1

Comprobación del paquete frente a las responsabilidades asignadas, el contrato técnico común y los
criterios de la rúbrica P2 que corresponden a esta parte.

---

## 1. Responsabilidades asignadas

| # | Responsabilidad | Estado | Dónde está |
|---|---|---|---|
| 1 | Revisar y completar únicamente lo necesario del EDA de la versión 2 | Cumplido | `01_baselines.ipynb`, secciones 2 y 3 |
| 2 | Implementar persistencia histórica | Cumplido | Sección 5 |
| 3 | Implementar regresión logística | Cumplido | Sección 6 |
| 4 | Preprocesamiento, escalamiento y ajuste de regularización | Cumplido | Secciones 4 y 6 |
| 5 | Resultados, curva PR, matriz de confusión y métricas | Cumplido | Secciones 7 y 8 |
| 6 | Redactar problema, datos, EDA y metodología de baselines | Cumplido | `paper/secciones_integrante1.md` |

### Detalle del punto 1

No se rehízo el EDA de P1. Se añadió solo lo que P2 requiere:

- Auditoría de calidad acotada a lo que condiciona la construcción de variables (divisiones por
  cero, intervalos negativos, continuidad de la cadena).
- Distribución de la posición relativa por clase y tabla de transición, que justifican el baseline
  de persistencia.
- Estacionalidad de calendario, que justifica las variables cíclicas.

### Detalle del punto 4

- Imputación por mediana y estandarización dentro de un `Pipeline`, ajustadas únicamente con
  entrenamiento.
- Búsqueda sobre penalización (L1 y L2), C ∈ {0,01; 0,1; 1; 10} y `class_weight` ∈ {None,
  balanced}: 16 configuraciones.
- Ajuste adicional del umbral de decisión sobre validación.

---

## 2. Contrato técnico común

Verificación automatizada sobre el código del notebook.

| Regla acordada | Estado |
|---|---|
| Mismo archivo fuente y filtro temporal de la versión 2 | Verificado contra el zip de P1 |
| Orden de bloques por `height` | Cumplido |
| Objetivo: percentil 75 móvil de 1 008 bloques, etiqueta al bloque siguiente | Cumplido |
| Partición cronológica 70/15/15 con fronteras comunes | Cumplido |
| Predictores solo con información hasta el bloque actual | Cumplido y verificado por perturbación |
| Imputadores y escaladores ajustados solo con entrenamiento | Cumplido |
| Validación para selección final, sin mirar test | Cumplido |
| Métrica principal: F1 de la clase positiva | Cumplido |
| Métricas secundarias: precisión, exhaustividad, AP/PR-AUC, ROC-AUC, matriz de confusión | Cumplido |
| `random_state = 42` | Cumplido |
| CSV con el esquema acordado | Cumplido |

### Correspondencia exacta con la partición de P1

| Partición | P1 entregado | Esta entrega |
|---|---|---|
| Entrenamiento | 293 104 — 24,65 % — 2016-01-07 a 2021-05-28 | idéntico |
| Validación | 62 808 — 21,47 % — 2021-05-28 a 2022-08-05 | idéntico |
| Prueba | 62 808 — 24,93 % — 2022-08-05 a 2023-10-06 | idéntico |

Los resultados son comparables con los notebooks de los demás integrantes.

### Esquema del CSV

`results/baselines_resultados.csv` tiene exactamente las columnas acordadas, en el mismo orden:

```
modelo, particion, f1, precision, recall, average_precision, roc_auc, umbral, hiperparametros
```

### Ausencia de fuga de datos

Cuatro medidas, todas presentes en el notebook:

1. Partición cronológica, no aleatoria.
2. Umbral de la clase estimado solo con bloques anteriores a cada fila.
3. Transformaciones del pipeline ajustadas solo con entrenamiento.
4. Variables de entrada que solo miran hacia atrás.

A esto se añade una **comprobación explícita**: se perturba la tasa del bloque siguiente y se
verifica que ninguna de las 22 variables de entrada cambia, mientras que la variable objetivo sí lo
hace. La celda está en la sección 4.1 y su salida es `True` en ambos casos.

Se incorpora además una purga de una observación entre particiones, correspondiente a la última
fila de entrenamiento, cuya etiqueta se construye mirando el primer bloque de validación.

### El conjunto de prueba se consulta una sola vez

La partición de prueba no aparece en la búsqueda de hiperparámetros ni en el ajuste del umbral.
Se evalúa en la sección 7, después de fijar todas las decisiones de modelado. El diagnóstico de la
transferencia del umbral que aparece a continuación calcula el umbral óptimo de prueba
**únicamente para cuantificar la pérdida**, y está explícitamente marcado como tal; no modifica el
modelo ni el umbral reportado.

---

## 3. Rúbrica P2 — criterios que cubre esta parte

### Criterio 1: Problema, datos y EDA (4 puntos)

| Requisito | Estado |
|---|---|
| Problema y unidad de análisis claros | Formulación E/T/P y sección de alcance que distingue bloque de transacción |
| Descripción real de las columnas | Tabla de las 13 variables con significado e interpretación de `median_fee_rate` |
| Hallazgos del EDA relevantes para la predicción | Tabla de transición (52,8 % / 15,5 %) y estacionalidad semanal (11,3 % / 32,4 %) |
| Visualizaciones interpretadas | Cada figura va seguida de su lectura; ninguna queda sin comentar |

### Criterio 2: Metodología y aplicación de ML (4 puntos) — parte correspondiente

| Requisito | Estado |
|---|---|
| Baseline de persistencia | Implementado y justificado con evidencia del EDA |
| Logística supervisada | Implementada con pipeline, búsqueda y ajuste de umbral |
| Limpieza, features y preprocesamiento justificados | 22 variables en 4 grupos, con tres decisiones argumentadas |

Los dos enfoques comparativos (Random Forest y boosting) corresponden a otro integrante.

### Criterio 3: Diseño experimental y evaluación (4 puntos) — parte correspondiente

| Requisito | Estado |
|---|---|
| Split cronológico | Cumplido |
| Selección de hiperparámetros sin test | Cumplido |
| Métricas consistentes | Las seis del contrato, en todas las particiones |
| Curvas y matrices pertinentes | Curva PR con el punto de operación de la persistencia, matriz de confusión, F1 frente al umbral |

### Criterio 4: Análisis e interpretación — aporte parcial

El análisis de errores por periodos y el estudio de ablación corresponden a otro integrante. Esta
entrega aporta:

- Interpretación de los coeficientes con su limitación por colinealidad.
- Diagnóstico de la transferencia del umbral.
- Lectura de la matriz de confusión (7 689 falsos positivos frente a 6 340 falsos negativos).

### Criterio 5: Comunicación y repositorio — aporte parcial

- Notebook ejecutado completo, sin errores ni avisos.
- Texto de las secciones listo para insertar.
- Lectura del dataset compartido `eda/dataset.csv` por ruta relativa, sin copias.
- `requirements.txt` con versiones fijadas.

---

## 4. Observaciones y puntos a resolver con el equipo

### Nombre del archivo

El plan del proyecto nombra este notebook `notebooks/02_baselines_logistic.ipynb` y reserva
`01_eda.ipynb` para el EDA consolidado de la versión 2. Aquí se entrega como `01_baselines.ipynb`.
**Conviene acordar la numeración antes de integrar**, para evitar que dos archivos distintos lleven
el prefijo 01.

### Dos recuentos del conjunto de datos

P1 reporta 419 727 pares bloque-siguiente; este notebook trabaja con 418 720 observaciones
etiquetables. La diferencia son las 1 007 primeras filas, donde la ventana de 1 008 bloques aún no
está completa y el umbral no está definido. Ambas cifras son correctas y describen momentos
distintos del mismo procedimiento. **Los demás integrantes deberían llegar al mismo 418 720**; si
alguno reporta otra cifra, hay una discrepancia de protocolo que resolver antes de comparar
resultados.

### La referencia a batir es la persistencia, no la logística

En prueba, la persistencia obtiene F1 de 0,5763 y la regresión logística 0,5704. Conviene que
quien entrene los modelos de ensamble lo sepa antes de empezar: el objetivo no es superar a la
logística sino a la persistencia.

### Parte de la señal de calendario procede de la definición de la clase

El umbral se calcula sobre una ventana de aproximadamente una semana, de modo que la
estacionalidad semanal de la red queda incorporada en la referencia. El modelo puede aprenderla sin
que corresponda a una propiedad del mercado. Está registrado en las limitaciones y conviene que
aparezca también en la discusión del paper.

### Técnicas que exceden el temario del curso

Dos elementos del notebook van más allá de lo visto en clase y conviene tenerlos comprendidos antes
de la exposición:

- **Precisión promedio (AP) y curva precisión-exhaustividad.** Se usan por el desbalance de clases.
- **Ajuste del umbral de decisión.** Se justifica porque el corte por defecto de 0,5 no optimiza la
  métrica declarada cuando las clases están desbalanceadas.

---

## 5. Reproducibilidad

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r fase2/integrante1/requirements.txt
```

El notebook lee `eda/dataset.csv` desde la raíz del repositorio (ruta relativa
`../../../eda/dataset.csv`). Ejecutar `fase2/integrante1/notebooks/01_baselines.ipynb` en orden con
`fase2/integrante1/notebooks/` como directorio de trabajo.

El notebook incluye las salidas de una ejecución completa. Tiempo aproximado: cuatro minutos,
dominado por las ocho configuraciones con solver `saga`.

Entorno comprobado: Python 3.12, scikit-learn 1.9, pandas 3.0. El `requirements.txt` fija versiones
compatibles con el entorno del equipo.
