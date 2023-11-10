"""
Wrapper class for metrics
"""

class MetricInterface:
    """ holds a dict of metrics to be ran on data """

    def __init__(self, metric_dict):
        """
        metric_dict: key-> name of metric: value-> function to get score
        """
        self.metric_dict = metric_dict
    
    def score(X, y):
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
