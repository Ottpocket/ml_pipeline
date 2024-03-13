# ML Pipeline
An attempt to write OOP code for ease in prototyping models and developing pipelines

# Usage Outline of Project
```
from ml_pipeline.xval import XValSubclass
from ml_pipeline.model import ModelSubclass
from ml_pipeline.data import DataSubclass

#Data Step
train_data = #load data from source
test_data = #load test data from source
data = DataSubclass(train=train_data, test=test_data, **args)

#Model Step
model = ModelSubclass()

#Cross validation
XValSubclass()
XValSubclass.cross_validate(model=model, data=data)

xval.cross_validate(model=model, data=data)

#Getting results
print(xval.get_run_scores())
print(xval.get_fold_scores())
print(xval.get_oof(raw=False))
print(len(xval.get_oof()))
```

# ml_pipeline Overview

1. xval: takes in data, and performs some type of cross validation.
2. model: creates models or model decorators that play nice with `xval`
3. data: decorators for data that ensure it plays nice with xval.

# Roadmap for future Development

1. `xval`: single model/multiple pred columns capabilities.
2. `xval`: multiple target values/multiple models capabilities
3. `xval`: multiple metrics on multiple outputs 

Specifically, `sample_scripts/testing/test_all.py` is the point of departure for testing the above 3 points in roadmap.  **Current TODO**: finish `sample_scripts/testing/1_simple.py`.
**Create ModelDecoratorMultiClass** subclass for multiclass prediction