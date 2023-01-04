import numpy as np 
import pandas as pd 
import os

def reduce_mem_usage(df, verbose=False):
    """ 
    copy pastaed from https://www.kaggle.com/code/gemartin/load-data-reduce-memory-usage/notebook.
    iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def to_bin(df, num_bins, binned_features, test = None):
    '''
    Inplace adds binned columns to df and test, if test dataframe is wanted.
    Old column 'feat' is binned as column 'feat_bin'.
    
    NOTE: This might result in less bins than requested due to the 
    some values in the original columns having duplicates.  
    
    ARGUMENTS
    ------------------
    df: (pd.DataFrame) dataframe to have binned columns added
    num_bins: (int) the min number of elements to fit into a bin
    binned_features: (list of columns) the columns to be binned
    test: (pd.DataFrame) the test set to have binned columns
    
    USAGE
    --------------
    1.  Want to bin the Latitude column in train and test
    >>> to_bin(train, 100, binned_features=['Latitude'], test=test)
    '''
    #Ensure that the number of bins requested < number unique in features
    errors = ""
    for feat in binned_features:
        nunique = df[feat].nunique()
        if num_bins >= nunique:
            errors += f"\nERROR: requested {num_bins} bins for {feat} which only has {nunique} bins."
    if errors != "":
        raise Exception(errors)
        
        
    for feat in binned_features:
        #Finding the bin edge
        bins = np.quantile(a=df[feat], q=np.linspace(start=0, stop=1, num=num_bins+1))
        bins = list(set(bins)) #Gets rid of bins with duplicate edges
        bins.sort()
        
        #creating the bins
        df[f'{feat}_bin'] = pd.cut(df[feat], bins, labels=False, include_lowest=True)
        if test is not None:
            test[f'{feat}_bin'] = pd.cut(test[feat], bins, labels=False, include_lowest=True)
