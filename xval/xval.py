"""
Splits data to be ingested by models
"""
from abc import ABC, abstractmethod
from ml_pipeline.record_keeper.record_keeper import RecordKeeper
from ml_pipeline.data_set.data_set import DataSet

class XVal():
    """ Completes an `N` run `K` fold crossval process


    """

    def __init__(self, 
                 split_mechanism, 
                 data: DataSet,
                 record_keeper:RecordKeeper,
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

    def cross_validate(self, model, metric, data:DataSet):
        """ `N` run `K` fold crossval process """

        #runs 
        for seed in self.runs:
            self.record_keeper.run_start()
            self.split_mechanism(n_splits=self.folds, random_state = seed)
            
            for tr_idx, val_idx in self.split_mechanism.split(data.get_index()):
                self.record_keeper.fold_start()
                train_data.set_idx(tr_idx = tr_idx, val_idx = val_idx)
                self.cross_validate_fold(model, metric, data)
                self.record_keeper.fold_end()

    def cross_validate_fold(self, 
                       model: ModelDecorator, 
                       metric: Metric,
                       data: DataSet
                       ):
        """ the logic for a single fold in a run of the crossval  
        
        Note: generating the data for the loop occurs elsewhere
        """
        #Initializing the fold
        model.init()

        #training 
        model.fit(data.get_fit_data()) 
        self.oof = model.validate(data.get_val_data())
        score = metric(self.oof, data.get_fold_targets() )

        #predicting
        if data.has_test_data():
            self.preds = model.predict(data.get_test_data())
