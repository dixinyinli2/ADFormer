# ADFormer: Aggregation Differential Transformer for Passenger Demand Forecasting

## Data
`\data` contains the geospatial files from <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page> and processed data, in which '_destination.pkl' means the number of trips ending in regions and '_origin.pkl' means the number of trips starting in regions. The raw data comes from the previous website. We only counted the demand quantity for each region in each time period, resulting in a data format that is convenient for further processing.

## Method
The implementation code of the model is located in the `\model` directory. `module.py` contains components needed by various models, while `ADFormer.py` is the main code of the model.

## Execute
`\utils` consists of `config.py`, which handles experiment configurations; `dataset.py`, which processes the `.pkl` files into the input format; `trainer.py`, which describes the training (evaluation) process; and `utils.py`, which contains utility functions needed for experiments.
For training a model, you can run:
```
python main.py

The trained model will be saved in the `\log` folder.

