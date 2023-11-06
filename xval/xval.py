"""
Splits data to be ingested by models
"""
from abc import ABC, abstractmethod
from ml_pipeline.record_keeper.record_keeper import RecordKeeper


class XVal():
    """ Completes an `N` run `K` fold crossval process


    """

    def __init__(self, 
                 split_mechanism, 
                 record_keeper:RecordKeeper,
                 runs:list=[42], 
                 folds:int=5, ):
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

    def cross_validate(self, model, metric, train_data, test_data=None):
        """ `N` run `K` fold crossval process """

        #runs 
        for seed in self.runs:
            self.record_keeper.run_start()
            self.split_mechanism(n_splits=self.folds, random_state = seed)
            
            for tr_idx, val_idx in self.split_mechanism.split(train):
                self.record_keeper.fold_start()
                self.cross_validate_fold(model, metric, train_data, tr_idx, val_idx, test_data)
                self.record_keeper.fold_end()

    def cross_validate_fold(self, 
                       model: ModelDecorator, 
                       metric: Metric,
                       train_data: DataPipeline,
                       tr_idx: list,
                       val_idx: list,
                       test_data: DataPipeline=None,
                       ):
        """ the logic for a single fold in a run of the crossval  
        
        Note: generating the data for the loop occurs elsewhere
        """
        #Initializing the fold
        model.init()
        train_data.set_idx(train = tr_idx, val = val_idx)

        #training 
        self.oof = model.fit(train_data) #work on this
        score = metric(train_data, self.oof)

        #predicting
        if test_data is not None:
            self.preds = model.predict(test_pipeline)
