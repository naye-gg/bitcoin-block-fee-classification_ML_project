# Entrega Integrante 1 — EDA, baselines y formulación

Proyecto Final de Machine Learning (CS3061) — UTEC 2026-2
**Etapa 2 (P2).** Predicción del nivel relativo de comisiones del siguiente bloque de Bitcoin.

Esta carpeta contiene la parte del proyecto correspondiente a EDA, modelos de referencia y
redacción de las secciones de problema, datos, EDA y metodología de baselines. Los modelos de
ensamble y el estudio de ablación corresponden a los otros integrantes.

## Contenido

```text
├── README.md                          este archivo
├── BITACORA.md                        decisiones del proyecto y su justificación
├── VERIFICACION.md                    comprobación frente al contrato y la rúbrica
├── requirements.txt
├── notebooks/
│   └── 01_baselines.ipynb             notebook ejecutado
├── paper/
│   └── secciones_integrante1.md       texto listo para insertar en el documento IEEE
└── results/
    ├── baselines_resultados.csv       métricas finales, esquema acordado con el equipo
    ├── busqueda_logistica.csv         16 configuraciones evaluadas sobre validación
    └── figures/                       5 figuras
```

## Resultado principal

| Modelo | Partición | F1 | Precisión | Exhaustividad | AP | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Persistencia | Validación | 0,5398 | 0,5405 | 0,5391 | — | — |
| Persistencia | Prueba | **0,5763** | 0,5770 | 0,5755 | — | — |
| Reg. logística | Validación | **0,5572** | 0,5157 | 0,6060 | 0,5929 | 0,8090 |
| Reg. logística | Prueba | 0,5704 | 0,5478 | 0,5950 | 0,6166 | 0,8061 |

La regresión logística supera a la persistencia en validación pero no en prueba. El modelo
discrimina (AP de 0,6166 sobre una tasa base de 0,2493), pero no supera a la inercia del mercado en
un punto de operación concreto. El diagnóstico descarta que se deba a la transferencia del umbral.

La referencia a batir por los modelos de ensamble es **la persistencia, con F1 de 0,5763 en
prueba**.

## Ejecutar

Desde la raíz del repositorio:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r fase2/integrante1/requirements.txt
```

El notebook lee el dataset compartido del repositorio, `eda/dataset.csv`, mediante la ruta relativa
`../../../eda/dataset.csv`; no hace falta copiarlo. Ejecutar `notebooks/01_baselines.ipynb` con
`fase2/integrante1/notebooks/` como directorio de trabajo. El notebook ya incluye las salidas de una
ejecución completa.
