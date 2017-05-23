# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 20:32:50 2017

@author: SyW
"""

import sys
import time
import csv
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras import regularizers


def read():
    #with open('DAX-price-2-years.csv', 'rb') as price_csv, open('DAX-sentiment-2-years.csv', 'rb') as sentiment_csv:
    with open('GLD_price.csv', 'rb') as price_csv, open('GLD_Sentiment.csv', 'rb') as sentiment_csv:
        price_reader = csv.reader(price_csv)
        sentiment_reader = csv.reader(sentiment_csv)

        trends = {}
        last_price = 0
        for row in price_reader:
            price = float(row[4])
            date = row[0]
            if last_price != 0:
                trends[date] = 1 if price > last_price else 0
            last_price = price
        m = len(trends.keys())  # total data

        sentiments = {}
        event_id = {}

        event_count = {}

        n = 0  # total characteristics
        for row in sentiment_reader:
            date = row[0]
            event = (int(row[1]), int(row[2]), int(row[3]))
            if event not in event_id:
                event_id[event] = n
                n += 1

                event_count[event] = 1
            else:
                event_count[event] += 1

            weight = int(row[4])
            if date not in sentiments:
                sentiments[date] = {}
            sentiments[date][event] = weight

    i = 0
    for event in event_count.keys():
        if event_count[event] < 200:
            event_id[event] = -1
            n -= 1
        else:
            event_id[event] = i
            i += 1

    return trends, sentiments, event_id, m, n


def build_vectors(trends, sentiments, event_id, m, n):
    X_all = np.zeros((m, n + 1))
    y_all = np.zeros(m)

    dates = sorted(trends.keys())
    for i in xrange(0, m):
        date = dates[i]
        for event in sentiments[date]:
            if event_id[event] != -1:
                X_all[i, event_id[event]] = sentiments[date][event]
            # X_all[i, event_id[event]] = sentiments[date][event]
        X_all[i, n] = 1
        y_all[i] = trends[date]

    return X_all, y_all


def train(X, y):
    m = X.shape[0]
    n = X.shape[1]
    model = Sequential()
    layer_1 = Dense(8, activation='sigmoid', input_dim=n, W_regularizer=regularizers.l2())
    layer_out = Dense(1, activation='sigmoid', W_regularizer=regularizers.l2())
    model.add(layer_1)
    model.add(Dropout(0.5))
    model.add(layer_out)
    model.compile(
        optimizer = SGD(lr=0.01),
        loss = 'binary_crossentropy',
        metrics = ['binary_accuracy']
    )
    
    t1 = time.clock()
    model.fit(X, y, nb_epoch=10000, batch_size=m, verbose=0)
    print 'Training time: %fs' % (time.clock() - t1)
    
    #print in_layer.get_weights()
    return model


def test(X, y, model):
    yh = model.predict(X)
    '''for yi in yh:
        print "%.9f " % yi,
    print ""'''
    total = y.shape[0]
    correct = 0
    for i in xrange(total):
        yh[i] = 1 if yh[i] > 0.5 else 0
        if yh[i] == y[i]:
            correct += 1
    print 'Total: %d, Correct: %d, Accuracy: %f' % (total, correct, float(correct) / total)


if __name__ == '__main__':
    training_num = 300
    validation_num = training_num + 100

    # read trends and sentiments from file
    trends, sentiments, event_id, m, n = read()

    # build vectors
    X_all, y_all = build_vectors(trends, sentiments, event_id, m, n)
    X_all = X_all.astype('int32')

    # work out theta
    X = X_all[0:training_num]
    y = y_all[0:training_num]
    '''Xt, yt = X, y
    for i in range(6):
        X = np.concatenate((X, Xt))
        y = np.concatenate((y, yt))'''

    # build model
    model = train(X, y)

    # validation
    X_test = X_all[training_num:validation_num]
    y_test = y_all[training_num:validation_num]
    test(X_test, y_test, model)

    # test
    X_test = X_all[validation_num:]
    y_test = y_all[validation_num:]
    test(X_test, y_test, model)
