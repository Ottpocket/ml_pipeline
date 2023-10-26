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
train_pipeline = DataSubclass(train_data)
test_pipeline = DataSubclass(test_data)

#Model Step
model = ModelSubclass()

#Cross validation
XValSubclass(model=model, train = train_pipeline, test = test_pipeline, **kwargs) #outputs scores and saves models
```

## ml_pipeline Overview

1. xval: takes in data, and performs some type of cross validation.
2. model: creates models or model decorators that play nice with `xval`
3. data: decorators for data that ensure it plays nice with xval.
