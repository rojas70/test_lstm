# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import numpy as np
import pickle
import random


def split_data(data, train_pos=0.7, val_pos=0.8, test_pos=1.0):
    """
    splits data to training, validation and testing parts
    """
    random.shuffle(data)
    
    num = len(data)
    train_pos = int(num * train_pos)
    val_pos = int(num * val_pos)
    
    train_data = data[:train_pos]
    val_data = data[train_pos:val_pos]
    test_data = data[val_pos:]

    return train_data, val_data, test_data


def rnn_data(data, time_steps, out_time_steps=1, labels=False):
    """
    creates new data frame based on previous observation
      * example:
        l = [1, 2, 3, 4, 5]
        time_steps = 2
        -> labels == False [[1, 2], [2, 3], [3, 4]]
        -> labels == True [3, 4, 5]
    """
    
    rnn_df = []
    for i in range(data.shape[0] - time_steps - (out_time_steps-1)):
        if labels:
            data_ = data[i + time_steps: i + time_steps + out_time_steps, 0:3] # True: hand(0:3)
            data_ = data_.reshape((out_time_steps,3))
            rnn_df.append(data_)
        else:
            data_ = data[i: i + time_steps, 0:6] # False: hand (0:3) + elbow (3:6)
            rnn_df.append(data_ if len(data_.shape) > 1 else [[item] for item in data_])

    return rnn_df


def prepare_seqs_data(seqs, time_steps, out_time_steps):
    """
    Given the number of `time_steps` and some data,
    prepares training, validation and test data for an lstm cell.
    """
    print('length of seqs:', len(seqs))
    
    seqs_x = []
    seqs_y = []
    for i, seq in enumerate(seqs):
        print('shape of seq:', seq.shape)
        
        seq_x = rnn_data(seq, time_steps, out_time_steps)
        seq_y = rnn_data(seq, time_steps, out_time_steps, labels=True)
        seqs_x += seq_x
        seqs_y += seq_y

    print("length of seqs_x and seqs_y:", len(seqs_x), len(seqs_y))
    
    seqs_x = np.array(seqs_x, dtype=np.float32)
    seqs_y = np.array(seqs_y, dtype=np.float32)
    print("shape of seqs_x and seqs_y:", seqs_x.shape, seqs_y.shape)
    
    return seqs_x, seqs_y


def generate_data(file_name, time_steps, out_time_steps):
    """generates data with based on a function func"""
    
    pkl_file = open(file_name,'rb')
    datasets = pickle.load(pkl_file, encoding='iso-8859-1')
    print('length of tasks:', len(datasets))

    seqs = []
    for i, task in enumerate(datasets):
        print('\nTask {0} has {1} seqs'.format(str(i), len(task)))
        
        for j, seq in enumerate(task):
            print('seq {0} has shape {1}'.format(str(j), seq.shape))
            seqs.append(seq)
    print('num of seqs:', len(seqs))

    train_seqs, val_seqs, test_seqs = split_data(seqs)
    
    print('\ntrain_seqs info:')
    train_x, train_y = prepare_seqs_data(train_seqs, time_steps, out_time_steps)
    print('\nval_seqs info:')
    val_x, val_y = prepare_seqs_data(val_seqs, time_steps, out_time_steps)
    print('\ntest_seqs info:')
    test_x, test_y = prepare_seqs_data(test_seqs, time_steps, out_time_steps)

    return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)
