# predictive-maintenance-lstm

## Overview
Determining the time available before a likely failure and being able to predict failures can help business’ 
better plan the use of their equipment, reduce operation costs, and avert issues before they become significant 
or catastrophic. The goal of predictive maintenance (PdM) is to allow for corrective actions and prevent 
unexpected equipment failure.

This project is a continuation of the work began as my project
[Spark ML](https://github.com/sabderra/predictive-maintenance-spark) for CSCI-E63 where
Spark (DataFrames, ML, Structured Streaming, etc) and Kafka were used to build an end-to-end workflow 
for predicting the Remaining Useful Life (RUL) of simulated turbofan engine data. A description of the data
can be found [here](https://github.com/sabderra/predictive-maintenance-spark/blob/master/README.md)

In this project we focus on using Keras and a Long Short-Term Memory (LSTM) based architecture to create an improved
 prediction model.
 
This repository includes a collection of notebooks and utility files.
* [data_analysis.ipynb](data_analysis.ipynb) - Loads and analyzes a sample of the data set.
* [train.ipynb](train.ipynb) - Prepares, transforms the data as well as building the model and training it.
* [model_prediction.ipynb](model_prediction.ipynb) - Runs preditions on the test data.

## Installing Dependencies
To install the packages: 
```bash
pip install -r requirements.txt 
```

Note some of the notebooks use tqdm_notebook for reporting progress, this requires
```bash
conda install -c conda-forge ipywidgets
```
If you don't want to bother with that replace tqdm_notebook with tqdm.

## Data

## Model

## Results

<p align="center">
<img src ="doc/images/test_predictions.png" />
</p>
