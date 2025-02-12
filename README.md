# ADFormer: Aggregation Differential Transformer for Passenger Demand Forecasting

## Data
`\data` contains the geospatial files from <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page> and processed data, in which '_destination.pkl' means the number of trips ending in regions and '_origin.pkl' means the number of trips starting in regions. The raw data comes from the previous website. We only counted the demand quantity for each region in each time period, resulting in a data format that is convenient for further processing.

## Method
The implementation code of the model is located in the `\model` directory. `module.py` contains components needed by various models, while `ADFormer.py` is the main code of the model.
