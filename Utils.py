import numpy as np 
import pandas as pd 
import os

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