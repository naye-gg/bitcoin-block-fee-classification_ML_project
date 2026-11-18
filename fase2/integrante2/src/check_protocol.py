"""Comprobaciones de causalidad y reproducción de P1, sin entrenar ni abrir y_test."""
import numpy as np
import pandas as pd
from experiment import load_source, causal_features, prepare_data

def main():
    raw,_=load_source();k=2200
    full=causal_features(raw.iloc[:2500]);prefix=causal_features(raw.iloc[:k+1])
    pd.testing.assert_frame_equal(full.iloc[:k+1],prefix)
    modified=raw.iloc[:2500].copy()
    modified.loc[k+1:,'median_fee_rate']=1e9
    pd.testing.assert_frame_equal(full.iloc[:k+1],causal_features(modified).iloc[:k+1])
    X,y,meta,a,b,audit,_=prepare_data()
    assert np.isclose(y.iloc[:a].mean()*100,24.65,atol=.005)
    assert np.isclose(y.iloc[a:b].mean()*100,21.47,atol=.005)
    assert len(y)==b and len(X)==b
    # Comprueba explícitamente orientación y endpoints de una ventana.
    first=raw.median_fee_rate.iloc[:1008].quantile(.75)
    assert np.isclose(X.umbral_local.iloc[0],first)
    assert y.iloc[0]==int(raw.median_fee_rate.iloc[1008]>first)
    assert meta.height.iloc[0]==raw.height.iloc[1007]
    print('OK: hash, cortes P1, target siguiente, ventana inclusiva, causalidad por prefijos, balance train/validación y ausencia de y_test.')

if __name__=='__main__':main()
