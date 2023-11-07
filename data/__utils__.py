"""
utility functions for data pipeline
"""
import pandas as pd

def check_for_features(df:pd.DataFrame, features:list, name:str):
    bad_feats = []
    for feat in features:
        if feat not in df.columns:
            bad_feats.append(feat)    
    if bad_feats != []:
        raise Exception(f'feats {bad_feats} not found in {name}.')