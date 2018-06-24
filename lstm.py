# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn as tflearn
from tensorflow.contrib import layers as tflayers

TIMESTEPS = 20
OUTPUT_TIMESTEPS = 5
OUTPUT_DIM = 3 # True: hand(0:3)


def lstm_model(num_units, rnn_layers, dense_layers=None, learning_rate=0.1, optimizer='Adagrad'):
    """
    Creates a deep model based on:
        * stacked lstm cells
        * an optional dense layers
    :param num_units: the size of the cells.
    :param rnn_layers: list of int or dict
                         * list of int: the steps used to instantiate the `BasicLSTMCell` cell
                         * list of dict: [{steps: int, keep_prob: int}, ...]
    :param dense_layers: list of nodes for each layer
    :return: the model definition
    """

    def lstm_cells(layers):
        cells = []
        
        if not isinstance(layers[0], dict):
            for step in layers:
                cell = tf.contrib.rnn.BasicLSTMCell(step, state_is_tuple=True)
                cells.append(cell)
            
        if isinstance(layers[0], dict):
            for layer in layers:
                cell = tf.contrib.rnn.BasicLSTMCell(layer['num_units'], state_is_tuple=True)
                if layer.get('keep_prob'):
                  cell = tf.contrib.rnn.DropoutWrapper(cell, layer['keep_prob'])
                cells.append(cell)
                
        return cells


    def dnn_layers(input_layers, layers):
        if layers and isinstance(layers, dict):
            return tflayers.stack(input_layers, 
                                  tflayers.fully_connected,
                                  layers['layers'],
                                  activation=layers.get('activation'),
                                  dropout=layers.get('dropout'))
        elif layers:
            return tflayers.stack(input_layers, tflayers.fully_connected, layers)
        else:
            return input_layers


    def _lstm_model(X, y):
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells(rnn_layers), state_is_tuple=True)
        
        X_ = tf.unstack(X, axis=1)
        print('X_ length:', len(X_))
        print('X_[0].shape:', X_[0].get_shape())
        
        output, layers = tf.contrib.rnn.static_rnn(stacked_lstm, X_, dtype=dtypes.float32)
        print('output shape of rnn:', output[-1].get_shape())
        
        y_pred = dnn_layers(output[-1], dense_layers)
        
        y_pred = tf.reshape(y_pred, [-1, OUTPUT_TIMESTEPS*OUTPUT_DIM])
        print('y_pred shape of dnn:', y_pred.get_shape())
        
        print('y shape:', y.get_shape())
        y = tf.reshape(y, [-1, OUTPUT_TIMESTEPS*OUTPUT_DIM])
        print('y shape:', y.get_shape())
        
        prediction, loss = tflearn.models.linear_regression(y_pred, y)
        
        train_op = tf.contrib.layers.optimize_loss(loss, tf.contrib.framework.get_global_step(), optimizer=optimizer, learning_rate=learning_rate)
            
        return prediction, loss, train_op

    return _lstm_model