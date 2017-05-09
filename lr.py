# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 23:41:26 2016

@author: WANG Shaoyang
"""

import csv
import numpy as np


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


def h(theta, x):
    return 1 / (1 + np.exp(-np.dot(x, theta) / theta.shape[0] * 0.0001))  # x can be a vector or a matrix


def train(X, y):
    m = X.shape[0]
    n = X.shape[1]
    print n, 'lalala'
    theta = np.zeros(n)
    alpha = 0.1

    for j in xrange(50000):
        #theta_ori = np.copy(theta)
        for i in xrange(m):
            theta += alpha * (y[i] - h(theta, X[i])) * X[i]
        #print np.sum(np.abs(theta_ori - theta))

    print '[' + ','.join([str(x) for x in theta]) + ']'
    return theta


def test(X, y, theta):
    total = y.shape[0]
    correct = 0
    for i in xrange(total):
        yh = h(theta, X[i])
        yh = 1 if yh > 0.5 else 0
        if yh == y[i]:
            correct += 1
    print 'Total: %d, Correct: %d, Accuracy: %f' % (total, correct, float(correct) / total)


if __name__ == '__main__':
    training_num = 300
    validation_num = training_num + 100

    # read trends and sentiments from file
    trends, sentiments, event_id, m, n = read()

    # build vectors
    X_all, y_all = build_vectors(trends, sentiments, event_id, m, n)

    # work out theta
    X = X_all[0:training_num]
    y = y_all[0:training_num]
    theta = train(X, y)

    # validation
    X_test = X_all[training_num:validation_num]
    y_test = y_all[training_num:validation_num]
    test(X_test, y_test, theta)

    # test
    X_test = X_all[validation_num:]
    y_test = y_all[validation_num:]
    test(X_test, y_test, theta)
