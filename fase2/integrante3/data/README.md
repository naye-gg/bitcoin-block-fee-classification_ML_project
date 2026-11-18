# Dataset de P1

Archivo requerido: `dataset.csv`, el mismo de P1 (810 909 filas, 13 columnas, hasta octubre de 2023).

SHA-256 esperado: `408620117782d1eb68a38eda4c2b2f6cdfc23ccedd6999ca8740d222a835ac6e` (ver `../../integrante2/PROTOCOLO.md`).

Para obtenerlo desde la raíz del repositorio:

```bash
git show main:eda/dataset.csv > fase2/integrante3/data/dataset.csv
```

Se usa `git show` en lugar de copiar `eda/dataset.csv` porque, en Windows con `core.autocrlf=true`, la copia de trabajo tiene saltos de línea CRLF y su hash no coincide con el del protocolo. El contenido guardado en git sí coincide.

El CSV está en `.gitignore` y no se sube al repositorio.
