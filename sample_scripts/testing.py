"""
Just testing to see if the system works or not
"""
import sys
sys.path.append('../..')
from ml_pipeline.xval.xval import XVal

class XValTestSubclass(XVal):
    """ A Subclass used only for `sample_scripts/testing.py` """

    def __init__(self):
        record_keeper = 
        metric_interface = 
        split_mechanism = 
        super().__init__((self,  
                 record_keeper,
                 metric_interface,
                 split_mechanism,
                 runs:list=[42], 
                 folds:int=5)