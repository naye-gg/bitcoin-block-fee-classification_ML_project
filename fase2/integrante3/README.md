# P2 — Robustez e interpretación (integrante 3)

Alvarado León, Adriana Celeste.

Análisis complementario a los modelos de integrante 2: *ablation study*, rendimiento por periodos, análisis de errores, limitaciones y estructura del informe IEEE. Se reutiliza sin cambios el protocolo de `../integrante2/PROTOCOLO.md` (target, particiones, gap y 33 características). El test queda reservado.

## Preguntas que responde

- **H2.** ¿Las variables de actividad de la red aportan información sobre las de comisiones? → *Ablation*: modelo A (comisiones) frente a modelo B (comisiones + actividad).
- **H3.** ¿El rendimiento se degrada en episodios de mayor variación de comisiones? → Evaluación segmentada por periodos y volatilidad.

## Archivos

- `notebooks/04_robustez.ipynb`: análisis completo.
- `src/comun.py`: importa la preparación de datos de integrante 2 y apunta al dataset de esta carpeta.
- `results/`: tablas CSV y `figures/`.
- `paper/`: secciones de discusión, limitaciones y conclusiones; borrador IEEE.
- `presentation/`: base de la presentación.
- `data/`: ver `data/README.md` para colocar el CSV.

## Ejecutar

Desde esta carpeta, con **Python 3.13** (en 3.14 no hay wheel de `numpy==2.2.6`; en Windows: `py -3.13 -m venv .venv`):

```bash
python -m venv .venv
source .venv/bin/activate        # Windows (Git Bash): source .venv/Scripts/activate
python -m pip install -r requirements.txt
```

Colocar el dataset según `data/README.md` y abrir `notebooks/04_robustez.ipynb`.
