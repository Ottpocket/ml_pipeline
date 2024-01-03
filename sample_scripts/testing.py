"""
Just testing to see if the system works or not
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
train = pd.read_csv('../sample_data/train.csv')
ancillary_train = pd.read_csv('../sample_data/ancillary_train.csv')
test = pd.read_csv('../sample_data/test.csv')
print(train.columns)
data = DataSetPandas(train, target='target', 
                     test=test,
                     features = ['feat1', 'feat2'],
                     ancillary_train= ancillary_train)

#Preparing dummy model
#model = ModelDecorator(OutputOnesModel)
model = ModelDecorator(HistGradientBoostingRegressor)


#Using XVal
xval = XValTestSubclass()
xval.cross_validate(model=model, data=data)
print(xval.get_run_scores())
print(xval.get_fold_scores())
print(xval.get_oof(raw=False))
print(len(xval.get_oof()))
