"""
Just testing to see if the system works or not
"""
from sklearn.model_selection import KFold
import pandas as pd

import sys
sys.path.append('../..')
from ml_pipeline.xval.xval import XVal
from ml_pipeline.record_keeper.record_keeper import RecordKeeper
from ml_pipeline.metric.metric import MetricInterfaceAcc
from ml_pipeline.model_decorator.model_decorator import ModelDecoratorOnes
from ml_pipeline.data.data_set import DataSetPandas

class XValTestSubclass(XVal):
    """ A Subclass used only for `sample_scripts/testing.py` """

    def __init__(self):
        record_keeper = RecordKeeper()
        metric_interface = MetricInterfaceAcc() 
        split_mechanism = KFold
        super().__init__(  
                 record_keeper,
                 metric_interface,
                 split_mechanism,
                 runs = [42], 
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

#Using XVal
xval = XValTestSubclass()
model = ModelDecoratorOnes()
xval.cross_validate(model=model, data=data)