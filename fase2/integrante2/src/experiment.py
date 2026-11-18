"""Experimento de integrante 2. No calcula métricas ni predicciones sobre test."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ml-p2-matplotlib')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
from pathlib import Path
import hashlib, json, time, platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (f1_score, precision_score, recall_score, average_precision_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay)
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SEED, WINDOW, GAP = 42, 1008, 1008
EXPECTED_SHA = '408620117782d1eb68a38eda4c2b2f6cdfc23ccedd6999ca8740d222a835ac6e'
RAW = ['size','tx_count','difficulty','median_fee_rate','avg_fee_rate','total_fees',
       'fee_range_min','fee_range_max','input_count','output_count','output_amount']
CONFIGS = {
 'RandomForest': [
  {'n_estimators':100,'max_depth':12,'min_samples_leaf':30},
  {'n_estimators':100,'max_depth':20,'min_samples_leaf':30},
  {'n_estimators':100,'max_depth':None,'min_samples_leaf':80}],
 'HistGradientBoosting': [
  {'max_iter':120,'max_leaf_nodes':15,'learning_rate':0.08,'min_samples_leaf':40,'l2_regularization':1.0},
  {'max_iter':180,'max_leaf_nodes':31,'learning_rate':0.06,'min_samples_leaf':80,'l2_regularization':5.0},
  {'max_iter':180,'max_leaf_nodes':15,'learning_rate':0.10,'min_samples_leaf':80,'l2_regularization':5.0}]
}

def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)

def load_source(path=None):
    path = Path(path) if path else ROOT/'data/dataset.csv'
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != EXPECTED_SHA:
        raise ValueError(f'Dataset distinto al de P1: {sha}')
    df = pd.read_csv(path).sort_values('height').reset_index(drop=True)
    assert df.shape == (810909,13)
    assert df.height.diff().dropna().eq(1).all()
    dates = pd.to_datetime(df.timestamp, unit='ms', utc=True)
    df = df.loc[dates.ge('2016-01-01')].reset_index(drop=True)
    assert len(df) == 419728 and df.height.diff().dropna().eq(1).all()
    return df, sha

def causal_features(df):
    """Cada fila h usa solo observaciones <= h; no consume etiquetas."""
    f = df.median_fee_rate
    X = df[RAW].astype(float).copy()
    X['umbral_local'] = f.rolling(WINDOW).quantile(.75)
    # +1 estabiliza cocientes con ceros, sin asignar una unidad física no verificada.
    X['fee_relativa'] = (f + 1)/(X.umbral_local + 1)
    X['ratio_avg_mediana'] = (df.avg_fee_rate + 1)/(f + 1)
    for lag in [1,6,144]:
        X[f'fee_lag_{lag}'] = f.shift(lag)
    for w in [6,144,1008]:
        X[f'fee_media_{w}'] = f.rolling(w).mean()
        X[f'fee_std_{w}'] = f.rolling(w).std(ddof=0)
    denom = df.tx_count.replace(0,np.nan)
    for col in ['size','input_count','output_count']:
        X[f'{col}_por_tx'] = df[col]/denom
    X['size_relativo_1008'] = df['size']/df['size'].rolling(WINDOW).max().replace(0,np.nan)
    dt = df.timestamp.diff()/1000
    X['intervalo_negativo'] = dt.lt(0).astype(float)
    X['intervalo_segundos'] = dt.where(dt.ge(0))
    dates = pd.to_datetime(df.timestamp,unit='ms',utc=True)
    hour = dates.dt.hour + dates.dt.minute/60
    for name, value, period in [('hora',hour,24),('dia_semana',dates.dt.dayofweek,7)]:
        X[f'{name}_sin'] = np.sin(2*np.pi*value/period)
        X[f'{name}_cos'] = np.cos(2*np.pi*value/period)
    return X.replace([np.inf,-np.inf],np.nan)

def prepare_data():
    df,sha = load_source()
    # Idéntica secuencia a P1: filtrar, shift(-1), descartar último,
    # rolling(1008), eliminar 1007 filas iniciales, recalcular cortes.
    current = df.iloc[:-1].copy()
    theta = current.median_fee_rate.rolling(WINDOW).quantile(.75)
    valid = theta.notna()
    meta = current.loc[valid,['height','timestamp']].reset_index(drop=True)
    n = len(meta); a,b = int(n*.70),int(n*.85)
    assert (n,a,b-a,n-b) == (418720,293104,62808,62808)
    X = causal_features(current).loc[valid].reset_index(drop=True)
    # El target solo se materializa para desarrollo. No se calcula y_test.
    target_dev = df.median_fee_rate.shift(-1).iloc[:-1].loc[valid].iloc[:b].to_numpy()
    ydev = pd.Series((target_dev > theta.loc[valid].iloc[:b].to_numpy()).astype('int8'))
    assert np.isfinite(X.iloc[:b].drop(columns=['intervalo_segundos']).to_numpy()).all()
    assert not any(c in X.columns for c in ['height','timestamp','tasa_siguiente','clase_movil'])
    audit=[]
    for name,start,end in [('train',0,a),('validacion',a,b),('test_reservado',b,n)]:
        rows=meta.iloc[start:end]
        audit.append({'particion':name,'filas':len(rows),'inicio_iloc':start,'fin_iloc_exclusivo':end,
          'height_inicial':int(rows.height.iloc[0]),'height_final':int(rows.height.iloc[-1]),
          'fecha_inicial_utc':str(pd.to_datetime(rows.timestamp.iloc[0],unit='ms',utc=True)),
          'fecha_final_utc':str(pd.to_datetime(rows.timestamp.iloc[-1],unit='ms',utc=True))})
    # Test: solo metadatos de fronteras. Se descartan sus predictores de esta ejecución.
    Xdev=X.iloc[:b].copy(); del X
    return Xdev,ydev,meta.iloc[:b],a,b,pd.DataFrame(audit),sha

def estimator(family,params):
    if family=='RandomForest':
        model=RandomForestClassifier(**params,random_state=SEED,n_jobs=4,
               max_features='sqrt',bootstrap=True,max_samples=.5)
    else:
        # Evita el holdout aleatorio interno; max_iter se selecciona por CV temporal.
        model=HistGradientBoostingClassifier(**params,random_state=SEED,early_stopping=False)
    return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),model)

def metrics(y,p,threshold):
    pred=(p>=threshold).astype(int)
    prec,rec,_=precision_recall_curve(y,p)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'f1':f1_score(y,pred,zero_division=0),
      'precision':precision_score(y,pred,zero_division=0),
      'recall':recall_score(y,pred,zero_division=0),
      'average_precision':average_precision_score(y,p),
      'pr_auc_trapezoidal':auc(rec,prec),
      'roc_auc':roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan,
      'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'n':len(y),'umbral':float(threshold)}

def search(X,y,train_end,out):
    folds=list(TimeSeriesSplit(n_splits=3,test_size=40000,gap=GAP).split(X.iloc[:train_end]))
    bounds=[]
    for k,(tr,va) in enumerate(folds,1):
        assert va[0]-tr[-1]-1==GAP
        bounds.append({'fold':k,'train_inicio':int(tr[0]),'train_fin':int(tr[-1]),
                       'val_inicio':int(va[0]),'val_fin':int(va[-1]),'gap':GAP})
    pd.DataFrame(bounds).to_csv(out/'folds.csv',index=False)
    records=[]; chosen={}; oof_store={}
    for family,configs in CONFIGS.items():
        candidate_scores=[]
        for j,params in enumerate(configs):
            pred_parts=[]
            for fold,(tr,va) in enumerate(folds,1):
                model=estimator(family,params); start=time.monotonic()
                with threadpool_limits(limits=4):
                    model.fit(X.iloc[tr],y.iloc[tr])
                    p=model.predict_proba(X.iloc[va])[:,1]
                m=metrics(y.iloc[va],p,.5)
                records.append({'modelo':family,'candidato':j,'fold':fold,
                  'segundos':time.monotonic()-start,'hiperparametros':json.dumps(params),**m})
                pred_parts.append(pd.DataFrame({'fila':va,'fold':fold,'y':y.iloc[va].to_numpy(),'p':p}))
                log(f'{family} candidato {j+1}/3 fold {fold}/3 F1={m["f1"]:.4f}')
                pd.DataFrame(records).to_csv(out/'busqueda_folds.csv',index=False)
            score=np.mean([r['f1'] for r in records if r['modelo']==family and r['candidato']==j])
            candidate_scores.append((score,j))
            oof_store[(family,j)]=pd.concat(pred_parts,ignore_index=True)
        # Desempate determinista: primer candidato.
        _,best=max(candidate_scores,key=lambda v:(v[0],-v[1]))
        oof=oof_store[(family,best)]
        thresholds=np.round(np.arange(.10,.901,.025),3)
        threshold_scores=[f1_score(oof.y,oof.p.ge(t),zero_division=0) for t in thresholds]
        ix=max(range(len(thresholds)),key=lambda i:(threshold_scores[i],-abs(thresholds[i]-.5)))
        threshold=float(thresholds[ix])
        pd.DataFrame({'umbral':thresholds,'f1_oof_ajuste':threshold_scores}).to_csv(out/f'umbrales_{family}.csv',index=False)
        oof.to_csv(out/f'oof_{family}.csv',index=False)
        chosen[family]={'params':configs[best],'candidato':best,'threshold':threshold,
                        'f1_cv_05':float(candidate_scores[best][0])}
    pd.DataFrame(records).groupby(['modelo','candidato']).agg(
        f1_media=('f1','mean'),f1_std=('f1','std'),ap_media=('average_precision','mean'),
        segundos=('segundos','sum')).reset_index().to_csv(out/'busqueda_resumen.csv',index=False)
    return chosen

def run():
    out=ROOT/'results';(out/'figures').mkdir(parents=True,exist_ok=True)
    (out/'manifest.json').unlink(missing_ok=True)  # no reutilizar una ejecución incompleta
    X,y,meta,a,b,audit,sha=prepare_data()
    audit.to_csv(out/'particiones.csv',index=False)
    pd.DataFrame({'feature':X.columns,'nulos_train':X.iloc[:a].isna().sum().values}).to_csv(out/'features.csv',index=False)
    log(f'Desarrollo: {a} train + {b-a} validación; {X.shape[1]} predictores; test reservado.')
    choices=search(X,y,a,out)
    (out/'seleccion.json').write_text(json.dumps(choices,indent=2))
    fit_end=a-GAP
    results=[];models={};preds={}
    for family,choice in choices.items():
        start=time.monotonic();model=estimator(family,choice['params'])
        log('Ajuste final '+family)
        with threadpool_limits(limits=4):
            model.fit(X.iloc[:fit_end],y.iloc[:fit_end])
            p_train=model.predict_proba(X.iloc[:fit_end])[:,1]
            p_val=model.predict_proba(X.iloc[a:b])[:,1]
        models[family]=model;preds[family]=p_val
        for part,yy,pp in [('train_ajuste',y.iloc[:fit_end],p_train),('validacion',y.iloc[a:b],p_val)]:
            results.append({'modelo':family,'particion':part,**metrics(yy,pp,choice['threshold']),
               'hiperparametros':json.dumps(model[-1].get_params(),sort_keys=True),
               'fit_filas':fit_end,'segundos_fit_y_predict':time.monotonic()-start})
        pd.DataFrame({'height':meta.height.iloc[a:b].to_numpy(),'y':y.iloc[a:b].to_numpy(),
                      'probabilidad':p_val,'prediccion':(p_val>=choice['threshold']).astype(int)}).to_csv(out/f'predicciones_validacion_{family}.csv',index=False)
        fig,ax=plt.subplots(figsize=(5,4));ConfusionMatrixDisplay.from_predictions(
            y.iloc[a:b],p_val>=choice['threshold'],display_labels=['No alta','Alta'],ax=ax,colorbar=False)
        ax.set_title(f'{family}\nValidación, umbral {choice["threshold"]:.3f}')
        fig.tight_layout();fig.savefig(out/'figures'/f'matriz_{family}.png',dpi=170);plt.close(fig)
        log(f'{family} validación F1={results[-1]["f1"]:.4f}, AP={results[-1]["average_precision"]:.4f}')
    table=pd.DataFrame(results);table.to_csv(out/'model_comparison.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,5))
    for family,p in preds.items():
        prec,rec,_=precision_recall_curve(y.iloc[a:b],p)
        ap=average_precision_score(y.iloc[a:b],p);ax.plot(rec,prec,label=f'{family}: AP={ap:.3f}')
    ax.axhline(y.iloc[a:b].mean(),color='gray',ls='--',label='Prevalencia en validación')
    ax.set(xlabel='Recall',ylabel='Precision',title='Curvas PR — validación cronológica',xlim=(0,1),ylim=(0,1.02))
    ax.legend();fig.tight_layout();fig.savefig(out/'figures/curvas_pr.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    names=list(choices);xs=np.arange(len(names));width=.35
    for offset,part in [(-width/2,'train_ajuste'),(width/2,'validacion')]:
        vals=[float(table.loc[(table.modelo==f)&(table.particion==part),'f1'].iloc[0]) for f in names]
        ax.bar(xs+offset,vals,width,label=part)
    ax.set_xticks(xs,names);ax.set(ylabel='F1 (umbral elegido en OOF)',ylim=(0,1),title='Brecha entrenamiento–validación')
    ax.legend();fig.tight_layout();fig.savefig(out/'figures/brecha_generalizacion.png',dpi=180);plt.close(fig)
    imp=pd.Series(models['RandomForest'][-1].feature_importances_,index=X.columns).sort_values(ascending=False)
    imp.rename_axis('feature').rename('importancia_mdi').to_csv(out/'importancia_rf_mdi.csv')
    fig,ax=plt.subplots(figsize=(8,6));imp.head(15).sort_values().plot.barh(ax=ax)
    ax.set(xlabel='Importancia por impureza (MDI)',title='Random Forest — no implica causalidad')
    fig.tight_layout();fig.savefig(out/'figures/importancia_rf_mdi.png',dpi=170);plt.close(fig)
    manifest={'dataset_sha256':sha,'source_commit':'7fbe2058c800eb5726f557b49e340bca4e338296',
      'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
      'scikit_learn':sklearn.__version__,'seed':SEED,'gap':GAP,'fit_rows':fit_end,
      'test_evaluado':False,'features':list(X.columns),'seleccion':choices,
      'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    from report import write_results
    write_results(ROOT)
    log('Terminado. Test no evaluado.');return table

if __name__=='__main__':run()
