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
        pass

    @abstractmethod
    def fold_end(self):
        pass
    