"""
Create a class to split data into folds
"""
from sklearn.model_selection import KFold
class SplitMechanism:
    """ base class to split data into folds 
    
    A shell to hold the actual splitting device, i.e. KFold.
    """

    def __init__(self, splitter):
        self.splitter = splitter

    def initialize_splitter(self, **kwargs):
        """ to be created every `XVal` object `run` """
        self.splitter(**kwargs)
        
    def split(self, X, y=None):
        if y is None:
            return self.splitter.split(X)
        else:
            return self.splitter.split(X, y)
        
        
class SplitMechanismKFold:
    """Uses KFolds """

    def __init__(self):
        super().__init__(splitter = KFold)

    def initialize_splitter(self, **kwargs):
        self.splitter(**kwargs)
        
    def split(self, X, y=None):
        if y is None:
            return self.splitter.split(X)
        else:
            return self.splitter.split(X, y)