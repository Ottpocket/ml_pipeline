"""
Basic means of tracking what is going on in the crossval process
"""
import time
from sklearn.metrics import accuracy_score
from ml_pipeline.record_keeper.__utils__ import print_block
class RecordKeeper:
    """ Base record keeping device: Only Keeps time elapsed. """
    def __init__(self):
        """ """
        #keep track of how long a fold/run takes
        self.run_start_time = time.time()
        self.fold_start_time = time.time() 

        self.fold_scores = {} #dict w/ keys as metric and values as scores
        self.run_scores = {} #dict w/ keys as metrics as values as scores

    def __add_metrics_to_records__(self, records, metrics):
        """ updates either `fold_scores` or    `run_scores` with new values"""
        
        #Updating the records
        for metric_name, metric_value in metrics.items():
            if metric_name in records.keys():
                records[metric_name].append(metric_value)
            else:
                records[metric_name] = [metric_value]
        
        #Checking to make sure the records all are identical length
        record_lengths = {}
        all_same_length = True
        lengths = -1
        for metric_name in records.keys():
            curr_len = len(records[metric_name])
            record_lengths[metric_name] = curr_len
            #initializing
            if lengths == -1:
                lengths = curr_len
            
            #ensuring all metrics have same # of values
            else:
                if curr_len != lengths:
                    all_same_length = False
        
        #One metric has more values than others.  An error:
        if not all_same_length:
            msg = f'''
            ERROR: some metrics have differing number of observations.  metrics:
            {record_lengths}
            '''
            raise Exception(msg)
        
    def fold_start(self):
        """ called at xval fold start """
        self.fold_start_time = time.time()
    
    def fold_end(self, fold_scores):
        """ called at xval fold end.  records metric scores"""
        fold_scores['time'] = time.time() - self.fold_start_time
        self.__add_metrics_to_records__(records=self.fold_scores, metrics = fold_scores)

    def run_start(self):
        """ called at xval run start """
        self.run_start_time = time.time()

    def run_end(self, run_score):
        """ called at xval run end """
        run_score['time'] = time.time() - self.run_start_time
        self.__add_metrics_to_records__(records=self.fold_scores, metrics = run_score)



        
        
class RecordKeeperPrint(RecordKeeper):
    """ Prints start of folds and runs.

    Uses `metric` objects to print metric information. 
    """

    def __init__(self):
        self.fold = 1
        self.run = 1
        super().__init__()

    def fold_start(self):
        """ called at xval fold start """
        print_block(heading= f'Starting fold {self.fold}',
                            size='small')
        super().fold_start()

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
