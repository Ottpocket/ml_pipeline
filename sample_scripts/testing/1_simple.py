"""
For Regression task, running a model over 
    -single target
    -single metric

To get the 
"""

from sklearn.model_selection import KFold
import pandas as pd
import sys
sys.path.append('../../..')
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
train = pd.read_csv('../../sample_data/march24/train.csv')
test =  pd.read_csv('../../sample_data/march24/test.csv')
ss =    pd.read_csv('../../sample_data/march24/sample_submission.csv')

#Initializing data object
features = ['X_Minimum','X_Maximum','Y_Minimum','Y_Maximum','Pixels_Areas','X_Perimeter','Y_Perimeter','Sum_of_Luminosity','Minimum_of_Luminosity','Maximum_of_Luminosity','Length_of_Conveyer','TypeOfSteel_A300','TypeOfSteel_A400','Steel_Plate_Thickness','Edges_Index','Empty_Index','Square_Index','Outside_X_Index','Edges_X_Index','Edges_Y_Index','Outside_Global_Index','LogOfAreas','Log_X_Index','Log_Y_Index','Orientation_Index','Luminosity_Index','SigmoidOfAreas']
target = 'Pastry'
data = DataSetPandas(train, target='Pastry',
                     test=test,
                     features = features)


#############################################
#############################################
#Regression task
#############################################
#############################################
#Initializing model object
model = ModelDecoratorMultiClass(HistGradientBoostingRegressor)

#Using XVal
xval = XValTestSubclass()
xval.cross_validate(model=model, data=data)
print(xval.get_run_scores())
print(xval.get_fold_scores())
print(len(xval.get_oof()))

#Saving predictions
ss[target] = xval.get_oof(raw=False)

#############################################
#############################################
#Classification task
#############################################
#############################################
#Initializing model object
model = ModelDecorator(HistGradientBoostingRegressor)