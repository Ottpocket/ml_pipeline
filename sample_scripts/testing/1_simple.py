"""
Runs a model over 
    -single target
    -single metric
"""
from sklearn.model_selection import KFold
import pandas as pd

import sys
sys.path.append('../..')
from ml_pipeline.xval.xval import XVal
from ml_pipeline.record_keeper.record_keeper import RecordKeeper
from ml_pipeline.metric.metric import MetricInterfaceMAE
from ml_pipeline.model_decorator.model_decorator import ModelDecorator#, OutputOnesModel
from ml_pipeline.data.data_set import DataSetPandas
from sklearn.ensemble import HistGradientBoostingRegressor

#Subclassed xval class
class XValTestSubclass(XVal):
    """ A Subclass used only for `sample_scripts/testing.py` """

    def __init__(self):
        record_keeper = RecordKeeper()
        metric_interface = MetricInterfaceMAE() 
        split_mechanism = KFold
        super().__init__(  
                 record_keeper,
                 metric_interface,
                 split_mechanism,
                 runs = [42, 43], 
                 folds = 5)
        
#Loading data
train = pd.read_csv('../sample_data/march24/train.csv')
test =  pd.read_csv('../sample_data/march24/test.csv')
ss =    pd.read_csv('../sample_data/march24/sample_submission.csv')
data = DataSetPandas(train, target='target', 
                     test=test,
                     features = ['feat1', 'feat2'],
                     ancillary_train= ancillary_train)