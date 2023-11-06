"""
Basic means of tracking what is going on in the crossval process
"""
import time
from abc import ABC
from ml_pipeline.record_keeper.__utils__ import print_block
class RecordKeeper(ABC):
    """ Base record keeping device: Only Keeps time elapsed. """
    def __init__(self):
        self.run_start_time = time.time()
        self.fold_start_time = time.time() 

        self.run_tot_time = 0.
        self.fold_tot_time = 0.

    def fold_start(self):
        """ called at xval fold start """
        self.fold_tot_time = 0.
        self.fold_start_time = time.time()

    
    def fold_end(self):
        """ called at xval fold end """
        self.fold_tot_time = time.time() - self.fold_start_time

    def run_start(self):
        """ called at xval run start """
        self.run_tot_time = 0.

    def run_end(self):
        """ called at xval run end """
        self.run_tot_time = time.time() - self.run_start_time
    
    def get_run_time(self):
        return self.run_tot_time
    
    def get_fold_time(self):
        return self.fold_tot_time

class RecordKeeperPrint(RecordKeeper):
    """ Prints start of folds and runs.

    Uses `metric` objecets to print metric information. 
    """

    def __init__(self):
        self.fold = 1
        self.run = 1
        super().__init__()

    def fold_start(self):
        """ called at xval fold start """
        print_block(heading= f'Starting fold {self.fold}',
                            size='small')

    def fold_end(self, metric=None):
        """ called at xval fold end """
        super().fold_end()
        print_block(heading= f'Ended fold {self.fold}',
                            metric = metric,
                            time = self.get_fold_time(),
                            size='small')
        self.fold +=1   

    def run_start(self):
        """ called at xval run start """
        print_block(heading= f'Starting run {self.run}',
                            size='big')

    def run_end(self, metric=None):
        """ called at xval run end """
        print_block(heading= f'Ended run {self.run}',
                            metric = metric,
                            time = self.get_run_time(),
                            size='big')
        self.fold=1
        self.run +=1
