import pandas as pd
import numpy as np
for name in ['train','ancillary_train', 'test']:
    df = pd.DataFrame({
        'feat1': np.random.uniform(size=100),
        'feat2': np.random.choice(['a','b','c'], size=100)
    })
    if name != 'test':
        df['target'] = np.random.uniform(size=100)
    df.to_csv(f'{name}.csv')