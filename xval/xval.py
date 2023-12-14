"""
Splits data to be ingested by models
"""
import numpy as np
from abc import ABC, abstractmethod
from ml_pipeline.record_keeper.record_keeper import RecordKeeper
from ml_pipeline.data.data_set import DataSet
from ml_pipeline.model_decorator.model_decorator import ModelDecorator
from ml_pipeline.split_mechanism.split_mechanism import SplitMechanism
from ml_pipeline.metric.metric import MetricInterface, MetricInterfaceTest

class XVal():
    """ Completes an `N` run `K` fold crossval process


    """
    def __init__(self,  
                 record_keeper:RecordKeeper,
                 metric_interface:MetricInterface=MetricInterfaceTest,
                 split_mechanism:SplitMechanism = None,
                 runs:list=[42], 
                 folds:int=5):
        """ 
        ARGUMENTS
        -----------------
        split_mechanism: class that splits data into folds
        record_keeper: device to take records of
        runs: (list of ints) random seeds to start a cross val split
            NOTE: the length of `runs` determines the number of times 
                    that the xval process is done
        fold: (int) # of folds per run
        """
        self.runs = runs
        self.folds = folds      
        self.split_mechanism = split_mechanism
        self.record_keeper = record_keeper
        self.metric_interface = metric_interface
        self.oof = []
        self.preds = [] #Just a placeholder.  This is a list of {runs} np arrays size {(num_test, num_classes)}

    def cross_validate(self, model:ModelDecorator, data:DataSet):
        """ `N` run `K` fold crossval process """
        self.oof = np.zeros(shape=(data.get_shape('train')[0], len(self.runs) )) 

        #runs 
        for run_num, seed in enumerate(self.runs):
            if data.has_test_data():
                num_cols = data.get_num_targets()
                self.preds.append(np.zeros( shape=(data.get_shape('test')[0], num_cols) )) 

            self.record_keeper.run_start()
            splitter = self.split_mechanism(n_splits=self.folds, shuffle=True, random_state = seed)
            model.update_run()
            print(f'Run: {run_num}')

            for tr_idx, val_idx in splitter.split(data.get_index()):
                print('fold, ', sep=' ')
                self.record_keeper.fold_start()
                data.set_index(tr_idx = tr_idx, val_idx = val_idx)
                score_dict = self.cross_validate_fold(model, data, run_num)
                self.record_keeper.fold_end(score_dict)

            run_score_dict = self.metric_interface.score(self.oof[:, run_num], data.get_targets() )
            self.record_keeper.run_end(run_score=run_score_dict)         


    def cross_validate_fold(self, 
                       model: ModelDecorator, 
                       data: DataSet,
                       run_num
                       ):
        """ the logic for a single fold in a run of the crossval  
        
        Note: generating the data for the loop occurs elsewhere
        """
        #Initializing the fold
        model.initialize_fold()

        #training
        model.fit(data.get_fit_data()) 
        self.oof[data.val_idx, run_num] = model.predict(data.get_val_data()[0])
        score_dict = self.metric_interface.score(self.oof[data.val_idx, run_num], data.get_fold_targets() )

        #predicting
        if data.has_test_data():
            self.preds[-1][:, run_num] += (model.predict(data.get_test_data()) /  self.folds)

        return score_dict
    
    def get_run_scores(self):
        return self.record_keeper.get_run_scores()
    def get_fold_scores(self):
        return self.record_keeper.get_fold_scores()
    
def get_predictions(self, raw=True):
    ''' Returns predictions
    
    Arguments
    --------------------
    raw: (bool) if True, gives the list of run predictions, 
                if False, takes the mean across the elements of list 
    '''
    if raw:
        return self.preds
    else:
        return sum(self.preds) / len(self.preds)
