"""Renderiza los resultados del paper a partir de los CSV, sin inventar cifras."""
from pathlib import Path
import json
import pandas as pd

def write_results(root):
    root=Path(root);out=root/'results'
    table=pd.read_csv(out/'model_comparison.csv')
    selected=json.loads((out/'seleccion.json').read_text())
    val=table[table.particion.eq('validacion')].set_index('modelo')
    train=table[table.particion.eq('train_ajuste')].set_index('modelo')
    rf=val.loc['RandomForest'];gb=val.loc['HistGradientBoosting']
    rows=[]
    for name,label in [('RandomForest','RF'),('HistGradientBoosting','HGB')]:
        r=val.loc[name]
        rows.append(f"{label} & {r.f1:.4f} & {r.precision:.4f} & {r.recall:.4f} & {r.average_precision:.4f} & {r.roc_auc:.4f} \\\\")
    latex=r'''\subsection{Resultados de los modelos de ensamble}
La Tabla~\ref{tab:ensambles-validacion} presenta los resultados de las dos familias sobre el mismo periodo de validación, posterior al entrenamiento. Estas cifras no corresponden a la prueba final.

\begin{table}[htbp]
\caption{Resultados en validación cronológica (62\,808 ejemplos). AP corresponde a average precision.}
\label{tab:ensambles-validacion}
\centering
\small
\begin{tabular}{lccccc}
\hline
Modelo & F1 & Prec. & Recall & AP & ROC-AUC \\
\hline
'''+ '\n'.join(rows)+r'''
\hline
\end{tabular}
\end{table}

'''
    latex+=f'Los umbrales de probabilidad seleccionados mediante predicciones fuera de fold fueron {rf.umbral:.3f} para RF y {gb.umbral:.3f} para HGB. El F1 en los datos de ajuste fue {train.loc["RandomForest","f1"]:.4f} y {train.loc["HistGradientBoosting","f1"]:.4f}, respectivamente. La diferencia entre ajuste y validación debe interpretarse considerando tanto la capacidad del modelo como los cambios temporales de los datos.\n\n'
    for name,label in [('RandomForest','RF'),('HistGradientBoosting','HGB')]:
        r=val.loc[name]
        latex+=f'En validación, {label} produjo {int(r.tp)} verdaderos positivos, {int(r.fp)} falsos positivos, {int(r.fn)} falsos negativos y {int(r.tn)} verdaderos negativos. '
    latex+='\n\n'
    latex+=f'HGB detectó {int(gb.tp-rf.tp)} casos positivos adicionales y produjo {int(gb.fp-rf.fp)} falsos positivos más que RF. Esto refleja el compromiso entre mayor recall y menor precisión observado con los umbrales fijados antes de evaluar este periodo.\n\n'
    diff=gb.f1-rf.f1
    latex+=f'La diferencia observada de F1 (HGB menos RF) fue {diff:+.4f}. Esta comparación describe un periodo y un conjunto acotado de configuraciones; no demuestra superioridad estadística ni universal. Las curvas precisión--recall y matrices de confusión complementan las métricas agregadas.\n\n'
    latex+=r'''La importancia por reducción de impureza de RF se interpreta de manera descriptiva, con cautela ante variables correlacionadas o de alta cardinalidad. No representa una estimación causal. HGB se interpreta mediante sus métricas y errores, sin atribuirle una propiedad de importancia que no expone. Las limitaciones incluyen dependencia temporal, ausencia de información del mempool y exposición previa a distribuciones del conjunto de prueba durante la reformulación de P1. La comparación conjunta con persistencia y regresión logística, así como la evaluación final en prueba, permanecen pendientes de integración. Por tanto, todavía no se concluye que los ensambles superen las referencias simples.
'''
    (root/'paper/resultados_modelos.tex').write_text('\n'.join(line.rstrip() for line in latex.splitlines())+'\n')
    summary=['RESULTADOS DE VALIDACIÓN — INTEGRANTE 2','',
      'No se evaluó test. Faltan integrar los baselines del integrante 1.', '']
    for name in val.index:
        r=val.loc[name]
        summary.append(f'{name}: F1={r.f1:.4f}; precision={r.precision:.4f}; recall={r.recall:.4f}; AP={r.average_precision:.4f}; ROC-AUC={r.roc_auc:.4f}; umbral={r.umbral:.3f}.')
        summary.append('Configuración: '+json.dumps(selected[name]['params']))
    summary+=['',f'Diferencia F1 HGB−RF: {diff:+.4f}. No equivale a significancia estadística.',
      'El código y PROTOCOLO.md fijan 33 predictores. Acordarlos con el equipo antes de comparar algoritmos.',
      'La partición de validación empieza en height 685293; test empieza en height 748101.',
      'Los modelos se ajustan hasta la fila 292095 de desarrollo (292096 ejemplos), con un gap de 1008.']
    (out/'RESUMEN.txt').write_text('\n'.join(summary)+'\n')
    print('Fragmento de resultados del paper actualizado desde CSV.')

if __name__=='__main__':write_results(Path(__file__).resolve().parents[1])
