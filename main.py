import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.contrib import learn

import lstm
from lstm import lstm_model
from data_processing import generate_data


## configuration of network architecture
TIMESTEPS = lstm.TIMESTEPS
OUTPUT_TIMESTEPS = lstm.OUTPUT_TIMESTEPS
OUTPUT_DIM = lstm.OUTPUT_DIM

RNN_LAYERS = [{'num_units': 50}]
DENSE_LAYERS = [OUTPUT_TIMESTEPS * OUTPUT_DIM]


## optimization hyper-parameters
TRAINING_STEPS = 10000
PRINT_STEPS = TRAINING_STEPS / 100
BATCH_SIZE = 100
LOG_DIR = './ops_logs/lstm'


## generate train/val/test datasets based on raw data
X, y = generate_data('./reg_fmt_datasets.pkl', TIMESTEPS, OUTPUT_TIMESTEPS)

## build the lstm model
model_fn = lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS)
estimator = learn.Estimator(model_fn = model_fn, model_dir = LOG_DIR)
regressor = learn.SKCompat(estimator)

## create a validation monitor
validation_monitor = learn.monitors.ValidationMonitor(X['val'], y['val'], every_n_steps=PRINT_STEPS)

## fit the train dataset
regressor.fit(X['train'], y['train'], monitors=[validation_monitor], batch_size=BATCH_SIZE, steps=TRAINING_STEPS)


## predict using test datasets
predicted = regressor.predict(X['test']) 
y_true = y['test'].reshape((-1, OUTPUT_TIMESTEPS * OUTPUT_DIM))
rmse = np.sqrt(((predicted - y_true) ** 2).mean())
print ("MSE: %f" % rmse)


## reshape for human-friendly illustration
y_true = y['test'].reshape((-1, OUTPUT_TIMESTEPS, OUTPUT_DIM))
predicted = predicted.reshape((-1, OUTPUT_TIMESTEPS, OUTPUT_DIM))

## illustrate some samples
idx = 3
print('X[test]', X['test'][:idx, :, :])

print('\n')
print('y[test]', y_true[:idx, :, :])

print('\n')
print('predicted', predicted[:idx, :, :])
