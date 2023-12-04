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
        self.preds = []

    def cross_validate(self, model:ModelDecorator, data:DataSet):
        """ `N` run `K` fold crossval process """
        self.oof = np.zeros(shape=(data.get_shape('train')[0], len(self.runs) ))
        if data.has_test_data():
            self.preds = np.zeros(shape=(data.get_shape('test')[0], len(self.runs) ) ) 

        #runs 
        for run_num, seed in enumerate(self.runs):
            self.record_keeper.run_start()
            splitter = self.split_mechanism(n_splits=self.folds, shuffle=True, random_state = seed)
            
            for tr_idx, val_idx in splitter.split(data.get_index()):
                self.record_keeper.fold_start()
                data.set_index(tr_idx = tr_idx, val_idx = val_idx)
                score_dict = self.cross_validate_fold(model, data)
                self.record_keeper.fold_end(score_dict)

            run_score_dict = self.metric.score(self.oof[:, run_num], data.get_targets() )
            self.record_keeper.run_end(score=run_score_dict)         


    def cross_validate_fold(self, 
                       model: ModelDecorator, 
                       data: DataSet
                       ):
        """ the logic for a single fold in a run of the crossval  
        
        Note: generating the data for the loop occurs elsewhere
        """
        #Initializing the fold
        model.initialize_fold()

        #training
        data2 = data.get_val_data() 
        print(len(data2))
        print(data2[0].shape)
        print(data2[1].shape)
        model.fit(data.get_fit_data()) 
        self.oof[data.val_idx, self.run] = model.predict(data.get_val_data())
        score_dict = self.metric.score(self.oof[data.val_idx, self.run], data.get_fold_targets() )

        #predicting
        if data.has_test_data():
            self.preds[:, self.runs] += (model.predict(data.get_test_data()) / self.runs)

        return score_dict
