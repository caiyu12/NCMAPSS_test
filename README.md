# Initial implementation of NCMAPSS dataset RUL prediction.
# Dataset
We chose the N-CMAPSS dataset as the validation dataset of our RUL prediction model. The dataset can be found at https://data.phmsociety.org/2021-phm-conference-data-challenge/.

We processed the data by applying the method mentioned in [Domain Adaptive Remaining Useful Life Prediction with Transformer](https://ieeexplore.ieee.org/document/9864208).
# Models
We adapted several time series forecasting models on RUL prediction. You can find them in the model.csv.
# Run the model
Edit the model.csv for the models that you want to validate, then run main.py.
You can find more information about the program in [hands_on.md](hands_on.md)

