"""Carga común para integrante 3, reutilizando el protocolo de integrante 2.

No redefine target, particiones ni características: importa
`fase2/integrante2/src/experiment.py` y solo cambia la ruta del dataset
para leer `fase2/integrante3/data/dataset.csv`. El test no se materializa.
"""
from pathlib import Path
import os
import sys
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'ml-p2-matplotlib'))

ROOT = Path(__file__).resolve().parents[1]
SRC_INTEGRANTE2 = ROOT.parent / 'integrante2' / 'src'
sys.path.insert(0, str(SRC_INTEGRANTE2))

import experiment  # noqa: E402  (código de integrante 2, sin modificar)

experiment.ROOT = ROOT  # load_source() lee ROOT/'data/dataset.csv' en tiempo de ejecución

SEED = experiment.SEED
GAP = experiment.GAP
WINDOW = experiment.WINDOW


def cargar_desarrollo():
    """Devuelve X, y, meta, a, b de train + validación (sin test).

    a = fin de train (exclusivo), b = fin de validación (exclusivo).
    Las comprobaciones de hash, tamaños y cortes de P1 se ejecutan dentro.
    """
    X, y, meta, a, b, audit, sha = experiment.prepare_data()
    return X, y, meta, a, b, audit
