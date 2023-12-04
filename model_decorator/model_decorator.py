"""
A decorator class for all models being ran by xval
"""
import numpy as np #only for dumb model

class ModelDecorator:
    """ Makes a model play nice with XVal class
    
    Takes in a model constructor, instantiates a new model each fold, 
    and saves it after training.  This class is intended to be used inside 
    the XVal.cross_validate method.
    
    NOTE
    --------------------------
    Saves models in `save_dir` as `model_1`, `model_2` etc...

    USAGE
    -----------------------
    1) decorating LGBM
        args = {lgbm_param_1:100, lgbm_param_2: 0.56}
        model = ModelDecorator(LGBMRegressor, model_args = args)

    PARAMETERS
    --------------------------------

     
    """
    def __init__(self, model_constructor, save_dir = 'saved_models', model_args={}):
        self.model_constructor = model_constructor
        self.save_dir = save_dir
        self.fold_num = 0
        self.model_args= model_args

    def initialize_fold(self):
        self.model_instance= self.model_constructor(**self.model_args)

    def fit(self, data):
        self.fold_num += 1
        X, y = data
        self.model_instance.fit(X, y)
        self.save()

    def save(self):
        save_path = f'{self.save_dir}/model_{self.fold_num}'
        self.model_instance.save(save_path)

    def predict(self, data):
        return self.model_instance.predict(data)
    
class OutputOnesModel:
    ''' a `model` that only predicts the value 1.
    
    Intended only as a dummy model for debugging/testing stuff
    '''
    def fit(self, X, y):
        pass
    
    def predict(self, data):
        return np.ones(shape= (data.shape[0],) )
    
    def save(self, **kwargs):
        pass
