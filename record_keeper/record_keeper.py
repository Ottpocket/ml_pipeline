"""
Basic means of tracking what is going on in the crossval process
"""
from abc import ABC, abstractmethod

class RecordKeeper(ABC):
    """ Base record keeping device """
    def __init__(self):
        pass

    @abstractmethod
    def fold_start(self):
        """ called at xval fold start """
        pass

    @abstractmethod
    def fold_end(self):
        """ called at xval fold end """
        pass

    @abstractmethod
    def run_start(self):
        """ called at xval run start """
        pass

    @abstractmethod
    def run_end(self):
        """ called at xval run end """
        pass
    

class RecordKeeperPrint(RecordKeeper):
    """ Prints start of folds and runs.

    Uses `metric` objecets to print metric information. 
    """

    def __init__(self):
        self.fold = 1
        self.run = 1

    def fold_start(self):
        """ called at xval fold start """
        print_block(heading= f'Starting fold {self.fold}',
                            size='small')

    def fold_end(self, metric=None):
        """ called at xval fold end """
        print_block(heading= f'Starting fold {self.fold}',
                            body = metric,
                            size='small')

    def run_start(self):
        """ called at xval run start """
        print_block(heading= f'Starting run {self.run}',
                            size='big')

    def run_end(self, metric=None):
        """ called at xval run end """
        print_block(heading= f'Starting run {self.run}',
                            body = metric,
                            size='big')
        self.fold=1
        self.run +=1
