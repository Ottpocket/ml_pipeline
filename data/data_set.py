"""
Classes that contain core logic of 
data 
"""
import pandas as pd
from abc import ABC, abstractmethod
from typing import Union
from __utils__ import check_for_features

class DataSet(ABC):
    """ abstract class """
    def __init__(self, train, test):
        """
        PARAMS
        ----------------
        train: train data
        test: test data
        """
        self.train = train
        self.test = test
        self.tr_idx = None
        self.val_idx = None 

    def set_index(self, tr_idx, val_idx):
        self.tr_idx = tr_idx
        self.val_idx = val_idx
    
    @abstractmethod
    def get_index(self):
        return []
    
    @abstractmethod
    def get_fold_targets(self):
        """ returns targets for a particular fold"""
        return []
    
    def get_fit_data(self):
        """ returns data for training a model on a fold"""
        return self.__get_data__(self.tr_idx)
    
    def get_val_data(self):
        """ returns data for validating a model on a fold"""
        return self.__get_data__(self.val_idx)
    
    @abstract
    def get_test_data(self):
        """ returns test data for predicting """
        return []
    
    @abstractmethod
    def __get_data__(self, idx):
        return []
    
    def has_test_data(self):
        return self.test is not None

class DataSetPandas(DataSet):
    """ pandas centric dataset """
    def __init__(self, train: pd.DataFrame,
                    target: str,
                    features: list,
                    test: Union[pd.DataFrame, None] = None,
                    ancillary_train: Union[pd.DataFrame, None] = None
                ):
        """
        PARAMS
        ---------------------
        train: train dataset
        target: target of training
        features: features for training
        test: test dataset
        ancillary_train: additional data used on every fold but not split or evaluated on
        """
        super().__init__(train, test)
        self.target = target
        self.ancillary_train = ancillary_train
        if features is None:
            self.features = [feat for feat in train.columns if feat != target]
        else:
            self.features = features

        #Sanity checks
        df_dict = {
            'train': self.train,
            'ancillary_train':self.ancillary_train,
            'test':self.test 
        }
        for name, df in df_dict.items():
            if df is not None:
                check_for_features(df, [self.target], name = name)            
                check_for_features(df, self.features, name = name)                    


    def __get_data__(self, idx):
        """ gets data from train dataset"""
        features = self.features
        target = self.target

        if self.ancillary_train is None:
            return self.train[features].iloc[idx], self.train[target].iloc[idx]
        else:
            tr_fold = pd.concat([
                self.train[features].iloc[idx],
                self.ancillary_train[features]
            ])
            tr_target = pd.concat([
                self.train[target].iloc[idx],
                self.ancillary_train[target]
            ])
            return tr_fold, tr_target
    
    def get_fold_targets(self):
        """ returns targets for a particular fold"""
        return self.train.loc[self.val_idx, self.target]

    def get_index(self):
        """ gets the train index"""
        return list(self.train.index)
    
    def get_test_data(self):
        """ returns test data for predicting """
        return self.test[self.features]
