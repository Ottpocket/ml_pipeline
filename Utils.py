import numpy as np 
import pandas as pd 
from sklearn.model_selection import StratifiedKFold, KFold

def likelihood_encoding(train, cat_col, encoding_cols, stats=['mean'], num_folds=5, test=None):
    '''
    Uses N-fold likelihood encoding to encode the categorical
    variable with the encoding_col's info.
    NOTE: This is inplace
    NOTE: indices for train and test must be 0-(nrows-1) or throws exception
    WARNING: for `count` statistic, use num_folds=1
    
    ARGUMENTS
    -----------------
    train: (pd.DataFrame)
    cat_col: (str) the name of the categorical column to be encoded.  
    encoding_cols: (list of str) the name of the column to be encoded.
    stats: (list of str) the statistic of the encoding_col to impute to the cat_col.
            NOTE: can include any `stat` compatable with df.agg({col:`stat`})
    num_folds: (int) number of folds to use for the encoding
    test: (pd.DataFrame) optional.  Encodes test data
    
    OUTPUT
    -------------------
    outputs a list of the names of the new columns
    NOTE: The new columns are already inplace added to the train (and test if specified)
    data.  The new columns are called cat_col_stat1_encoding_col1, ... , cat_col_statM_encoding_colN
    '''
    if type(train.index) is not pd.core.indexes.range.RangeIndex:
        error = f'''\ntrain must have a RangeIndex from 0 to {train.shape[0] -1}.
        \ntrain's index is instead {type(train.index)}.'''
        raise Exception(error)
    if test is not None:
        if type(test.index) is not pd.core.indexes.range.RangeIndex:
            error = f'''\ntest must have a RangeIndex from 0 to {test.shape[0]-1}.
            \ntest's index is instead {type(test.index)}.'''
            raise Exception(error)
    if cat_col in encoding_cols:
        raise Exception(f'{cat_col} found in encoding cols.')
    if 'count' in stats:
        warning = f'''`count` found in stats.  To prevent redundancies, 
        the count encoded column will be labelled as `{cat_col}_count_`'''
        print(warning)
        count_flag =True
        stats.remove('count')
    else:
        count_flag = False
    
    
    #Getting the correct aggregation dict for the groupby.agg(dict) 
    agg_dict = {}
    for encoding_col in encoding_cols:
        agg_dict[encoding_col] = stats
    upcaster = Upcaster(train) #upcasts data to prevent overflow
    upcaster.upcast(train, encoding_cols)
    
    def get_le_cols(train_df, new_df, agg_dict=agg_dict, cat_col=cat_col, train_idx=None, test_idx=None):
        '''
        train_df: (pd.DataFrame) df which has the raw features for statistics
        new_df: (pd.DataFrame) df which has statistics applied to it
        '''
        #use all indices if no subset provided
        if train_idx is None:
            train_idx = range(len(train_df)) 
        if test_idx is None:
            test_idx = range(len(new_df))

        agg = train_df.iloc[train_idx].groupby(cat_col).agg(agg_dict)
        reduce_mem_usage(agg)
        agg.columns = [f'{cat_col}_{tup[1]}_{tup[0]}' for tup in agg.columns]
        new_cols = new_df[[cat_col]].iloc[test_idx].merge(agg, on=cat_col, how='left')
        del new_cols[cat_col]

        #Mean impute NA values
        for col in agg.columns:
            nan_msk = new_cols[col].isnull()
            if np.sum(nan_msk) >0:
                stat_feat = col.split(cat_col)[-1].split('_')
                statistic = stat_feat[1]
                feat_col = '_'.join(stat_feat[2:])
                if statistic != 'count':
                    new_cols.loc[nan_msk, col] = train[feat_col].agg(statistic)
                else:
                    new_cols.loc[nan_msk, col] = 0

        new_cols.index = test_idx
        return new_cols
    
    #######################
    #Train le
    #######################
    if num_folds == 1:
        new_features = get_le_cols(train_df = train, new_df = train)
    
    else:
        #Kfold splitting to prevent leakage
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
        new_feature_list = []
        for i, (train_idx, test_idx) in enumerate(skf.split(train, train[cat_col])):
            new_cols = get_le_cols(train_df = train, new_df = train, 
                                   agg_dict=agg_dict, cat_col=cat_col, 
                                   train_idx=train_idx, test_idx=test_idx)
            new_feature_list.append(new_cols)


        new_features = pd.concat(new_feature_list)
        new_features.sort_index(inplace=True)

    for feat in new_features.columns:
        train[feat] = new_features[feat]
    le_col_names = new_features.columns
    
    #######################
    #test le
    #######################
    if test is not None:
        new_features = get_le_cols(train_df = train, new_df = test)
        for feat in new_features.columns:
            test[feat] = new_features[feat]
    
    #######################
    #count stat loose ends
    #######################
    if count_flag:
        count_col_name = f'{cat_col}_count_'
        count_dict = {encoding_cols[0]: ['count']}
        
        new_features = get_le_cols(train_df = train, new_df = train, agg_dict=count_dict)
        created_col_name = new_features.columns[0]
        train[count_col_name] = new_features[created_col_name]
        le_col_names.append(count_col_name)
        
        if test is not None:
            new_features = get_le_cols(train_df = train, new_df = test, agg_dict=count_dict)
            test[count_col_name] = new_features[created_col_name]
    upcaster.revert(train)#downcasts data to original dtype
    return list(le_col_names)
    
def reduce_mem_usage(df, verbose=False, obj_to_cat=True):
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
            if obj_to_cat:
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

            
class Upcaster:
    '''
    Upcasts columns and can revert to original state
    '''
    def __init__(self, df):
        self.dtype_dict = {}
        for col in df:
            self.dtype_dict[col] = df[col].dtype.name
        self.cols = None
    
    def upcast(self, df, cols):
        '''
        cols: (list of col names) columns to be upcasted
        '''
        for col in cols:
            if col not in self.dtype_dict.keys():
                raise exception(f'{col} not found in dtype dictionary.')
            
            if 'float' in self.dtype_dict[col]:
                if df[col].dtype.name != 'float64':
                    df[col] = df[col].astype('float64')
            elif 'int' in self.dtype_dict[col]:
                if df[col].dtype.name != 'int64':
                    df[col] = df[col].astype('int64')
            else:
                raise exception(f'{col} is neither `int` nor `float`.  It is {df[col].dtype.name}.')
        self.cols= cols
        
    def revert(self, df):
        if self.cols is None:
            raise Exception(f'No columns to revert.  Must call upcast method before reverting.')
        
        for col in self.cols:
            if col in ['int64', 'float64']:
                pass
            else:
                df[col] = df[col].astype(self.dtype_dict[col])

def xval(feats, train, metric, model, target, n_splits=5, test=None, extra = None, verbose=False):
    '''
    Crossvalidates the data using kfolds.  If test data provided, gives the test predictions
    averaged across folds.  
    If extra data given, it adds the extra data for each fold.
    
    ARGUMENTS
    ----------------------
    feats: (list of str) list of pandas colnames to use as features
    train: (pd.DataFrame) train data
    metric: (sklearn.metrics) the metric to use. i.e. roc_auc_score from sklearn.metrics
    model: (sklearn compatable model) the model to train and predict
    n_splits: (int) the number of folds for the cross val
    test: (pd.DataFrame) test data.  If provided, the trained models from each
                        fold make predictions on this data.  In place.
    extra: (pd.DataFrame) extra data to be used on every fold.  optional
    verbose: (bool) display outputs?
    
    OUTPUTS
    -----------------
    score: (float) the crossvalidation score.
    '''
    scores=[]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    if verbose:
        print(f'5: ',end='')
    for i, (train_idx, test_idx) in enumerate(kf.split(train)):
        if verbose:
            print(f'{i+1},',end='')
        if extra is None:
            model.fit(train[feats].iloc[train_idx], train[target].iloc[train_idx])
        else:
            temp_tr = pd.concat([train[feats].iloc[train_idx], extra[feats]])
            temp_targ = pd.concat([train[target].iloc[train_idx], extra[target]])
            model.fit(temp_tr, temp_targ)
        
        preds = model.predict(train[feats].iloc[test_idx])
        scores.append(metric(train[target].iloc[test_idx], preds))
        if test is not None:
            ss[target] += model.predict(test[feats]) / n_splits
    
    if verbose:
        print()
        print(scores)
        print(np.mean(scores))
    return np.mean(scores)
