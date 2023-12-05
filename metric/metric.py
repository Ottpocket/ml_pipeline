"""
Wrapper class for metrics
"""
from sklearn.metrics import accuracy_score, mean_absolute_error
class MetricInterface:
    """ holds a dict of metrics to be ran on data 
    
    Intended to work with the xval class

    EXAMPLE USAGE
    ------------------------------
    1) want to measure acc and AUC
        metric_interface = MetricInterface({'auc':sklearn.metrics.AUC, 'acc':sklearn.metrics.accuracy_score})
    """

    def __init__(self, metric_dict):
        """
        metric_dict: key-> name of metric: value-> function to get score
        """
        self.metric_dict = metric_dict
    
    def score(self, X, y):
        """ getting scores from the various metrics """
        score_dict = {}
        for name, metric_function in self.metric_dict.items():
            score_dict[name] = metric_function(X,y)
        return score_dict
    
class MetricInterfaceTest(MetricInterface):
    """ Just has a placeholder"""
    def __init__(self):
        def placeholder(X,y):
            return -1
        
        metric_dict = {'placeholder': placeholder}
        super().__init__(metric_dict=metric_dict)


class MetricInterfaceAcc(MetricInterface):
    def __init__(self):
        super().__init__({'acc':accuracy_score})

class MetricInterfaceMAE(MetricInterface):
    def __init__(self):
        super().__init__({'mae':mean_absolute_error})