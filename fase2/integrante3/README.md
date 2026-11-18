# P2 — Robustez e interpretación (integrante 3)

Alvarado León, Adriana Celeste.

Análisis complementario a los modelos de integrante 2: *ablation study*, rendimiento por periodos, análisis de errores, limitaciones y estructura del informe IEEE. Se reutiliza sin cambios el protocolo de `../integrante2/PROTOCOLO.md` (target, particiones, gap y 33 características). Este análisis no usa el test; la evaluación final en prueba está en `../integrante2/results/test_final/`.

## Preguntas que responde

- **H2.** ¿Las variables de actividad de la red aportan información sobre las de comisiones? → *Ablation*: modelo A (comisiones) frente a modelo B (comisiones + actividad).
- **H3.** ¿El rendimiento se degrada en episodios de mayor variación de comisiones? → Evaluación segmentada por periodos y volatilidad.

## Resultados principales (validación, mayo 2021 – agosto 2022)

| Conjunto | Variables | AP | F1 |
|---|---|---|---|
| A0: tasa actual | 3 | 0.535 | 0.539 |
| A: comisiones | 16 | 0.622 | 0.569 |
| B: + actividad de la red | 29 | 0.629 | 0.565 |
| C: + calendario (las 33) | 33 | 0.630 | 0.571 |
| Persistencia (referencia) | — | — | 0.540 |

- **H1:** la tasa actual sola equivale a la persistencia; la historia de comisiones aporta +0.088 de AP (IC 95 % [0.075, 0.099], bootstrap por bloques semanales).
- **H2:** la actividad de la red, incluidas las comisiones totales como en la H2 de P1, aporta +0.007 de AP (IC [0.003, 0.011]); el calendario no cambia la AP. En F1 las diferencias entre A, B y C son pequeñas y dependen del umbral.
- **H3:** degradación moderada en el tercil de mayor volatilidad (F1 0.554 frente a 0.567–0.582). El error principal son los cambios de clase: el modelo anticipa el 35 % de los finales de episodio y solo el 19 % de los inicios.

## Archivos

- `notebooks/04_robustez.ipynb`: análisis completo con interpretación, limitaciones y conclusiones.
- `src/comun.py`: importa la preparación de datos de integrante 2 y apunta al dataset de esta carpeta.
- `src/robustez.py`: ablation, bootstrap, análisis por periodos y volatilidad, análisis de errores y figuras.
- `results/`: tablas CSV, predicciones de validación, `manifest.json` y `figures/`.
- `paper/`: `robustez.tex` (resultados), `discusion.tex` (incluye consideraciones éticas), `limitaciones.tex` y `conclusiones.tex`.
- El artículo completo está en `../../paper/main.tex` y la presentación en `../../presentation/slides.tex`.
- `data/`: ver `data/README.md` para colocar el CSV.

## Ejecutar

Desde esta carpeta, con **Python 3.13** (en 3.14 no hay wheel de `numpy==2.2.6`; en Windows: `py -3.13 -m venv .venv`):

```bash
python -m venv .venv
source .venv/bin/activate        # Windows (Git Bash): source .venv/Scripts/activate
python -m pip install -r requirements.txt
```

Colocar el dataset según `data/README.md` y ejecutar:

```bash
python src/robustez.py
```

Tarda unos 3 minutos (16 ajustes y 4000 remuestreos de bootstrap) y sobrescribe `results/`. También se puede abrir `notebooks/04_robustez.ipynb`: muestra los resultados guardados y, con `REENTRENAR = True`, repite el experimento. El test no se utiliza.
