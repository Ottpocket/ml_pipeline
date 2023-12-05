'''
Creates `train.csv`, `ancillary_train.csv`, and `test.csv`.  
For train and ancillary_train, the target is a linear combination of its continuous variables.  
'''
import pandas as pd
import numpy as np
for name in ['train','ancillary_train', 'test']:
    df = pd.DataFrame({
        'feat1': np.random.uniform(size=100),
        'feat2': np.random.uniform(size=100),
        'feat3': np.random.choice(['a','b','c'], size=100)
    })
    if name != 'test':
        df['target'] = df['feat1'] + .3*df['feat2'] + np.random.uniform(size=100) * .1
    df.to_csv(f'{name}.csv')