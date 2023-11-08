"""
A decorator class for all models being ran by xval
"""

class ModelDecorator:
    """ generic decorator for models
    
    Saves models in `save_dir` as `model_1`, `model_2` etc...
    """
    def __init__(self, model, save_dir = 'saved_models'):
        self.model = model
        self.save_path = save_dict
        self.fold_num = 0

    def initialize_fold(self):
        pass

    def fit(self, data):
        self.fold_num += 1
        X, y = data
        self.model.fit(X, y)
        self.save()

    def save(self):
        save_path = f'{save_path}/model_{self.fold_num}'
        self.model.save(save_path)

    def predict(self, data):
        return self.model.predict(data)
    

