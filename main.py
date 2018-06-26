import numpy as np
import matplotlib
import csv
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

RNN_LAYERS = [{'num_units': 50}] #input 20*3*(50)
DENSE_LAYERS = [OUTPUT_TIMESTEPS * OUTPUT_DIM] #output: 5*3*(50)


## optimization hyper-parameters
TRAINING_STEPS = 100
PRINT_STEPS = TRAINING_STEPS / 100
BATCH_SIZE = 100
LOG_DIR = './ops_logs/lstm/'+str(TIMESTEPS)+str(OUTPUT_TIMESTEPS)

# MODEL_SAVED_PATH = './snapshot/model.ckpt'
# MODEL_SAVED_DIR = './snapshot'

def mse(pred, true):
    return np.sqrt(((pred - true) ** 2).mean())


## build the lstm model
model_fn = lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS)

## generate train/val/test datasets based on raw data
X, y = generate_data('./reg_fmt_datasets.pkl', TIMESTEPS, OUTPUT_TIMESTEPS)

estimator = learn.Estimator(model_fn = model_fn, model_dir = LOG_DIR)
# estimator = tf.estimator.Estimator(model_fn = model_fn, model_dir = LOG_DIR)
regressor = learn.SKCompat(estimator)

## create a validation monitor
validation_monitor = learn.monitors.ValidationMonitor(X['val'], y['val'], every_n_steps=PRINT_STEPS)

## fit the train dataset
regressor.fit(X['train'], y['train'], monitors=[validation_monitor], batch_size=BATCH_SIZE, steps=TRAINING_STEPS)

#prepare for testing
step = 0.05
ratio = [step * i for i in range(1, int(1 / step) + 1)]
mse_list = [0] * len(ratio)

#start testing
for X_test,y_test in zip(X['test'], y['test']):
    ## predict using test datasets
    predicted = regressor.predict(X_test)
    print('predicted:', predicted)
    y_true = y_test.reshape((-1, OUTPUT_TIMESTEPS * OUTPUT_DIM))
    print('y_true:', y_true)
    rmse = mse(predicted, y_true)
    print("MSE: %f" % rmse)
    ## reshape for human-friendly illustration
    y_true = y_test.reshape((-1, OUTPUT_TIMESTEPS, OUTPUT_DIM))
    predicted = predicted.reshape((-1, OUTPUT_TIMESTEPS, OUTPUT_DIM))

    # ## illustrate some samples
    # idx = 3
    # print('X[test]', X_test[:idx, :, :])
    #
    # print('\n')
    # print('y[test]', y_true[:idx, :, :])
    #
    # print('\n')
    # print('predicted', predicted[:idx, :, :])

    ## write mse against ratio to csv file
    for i,r in enumerate(ratio):
        index = int(r*len(y_true)-1)
        imse = mse(predicted[index],y_true[index])
        mse_list[i] = mse_list[i]+imse

mse_list[:]=[x/len(X['test']) for x in mse_list]

#print average result
print('ratio:', ratio)
print('mse:', mse_list)

#write to csv file
CSV_PATH = './csv/'+str(TIMESTEPS)+str(OUTPUT_TIMESTEPS)
with open(CSV_PATH+'result.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar=',', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(ratio)
    spamwriter.writerow(mse_list)




