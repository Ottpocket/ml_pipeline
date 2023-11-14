import sys
sys.path.append('../..')
from ml_pipeline.record_keeper.record_keeper import RecordKeeperPrint

rk = RecordKeeperPrint()
for run_num, seed in enumerate(range(2)):
            rk.run_start()
            
            for _ in range(3):
                rk.fold_start()
                pass
                rk.fold_end()
            rk.run_end()
