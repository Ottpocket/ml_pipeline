"""
Splits data to be ingested by models
"""

from abc import ABC, abstractmethod
from ml_pipeline.record_keeper.record_keeper import RecordKeeper


class XVal(ABC):
    """ base cross val class """

    def __init__(self):
        self.oof = []
        self.preds = []

    def cross_validate(self, record_keeper: RecordKeeper, 
                       model: ModelDecorator, 
                       metric: Metric,
                       train_data: DataPipeline,
                       tr_idx: list,
                       val_idx: list,
                       test_data: DataPipeline=None,
                       ):
        """ the logic for the inside of a crossval loop 
        
        Note: generating the data for the loop occurs elsewhere
        """
        #Initializing the fold
        record_keeper.fold_start()
        model.init()
        train_data.set_idx(train = tr_idx, val = val_idx)

        #training 
        record_keeper.train_start()
        self.oof = model.fit(train_data) #work on this
        score = metric(train_data, self.oof)
        record_keeper.train_end()

        #predicting
        if test_data is not None:
            record_keeper.pred_start()
            self.preds = model.predict(test_pipeline)
            record_keeper.pred_end()
