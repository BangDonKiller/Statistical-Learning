#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pandas

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


def gen_train_test_data():
    test_ratio = 0.3
    df = pandas.read_csv("./yelp_labelled.txt", sep="\t", header=None, encoding="utf-8")
    count_vect = CountVectorizer()
    X = count_vect.fit_transform(df[0])
    y = df[1].tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=0
    )
    print(X_train)
    return X_train, X_test, y_train, y_test


def multinomial_nb(X_train, X_test, y_train, y_test):
    # TODO: fill this function
    # train by X_train and y_train
    # report the predicting accuracy for both the training and the test data

    pass


def bernoulli_nb(X_train, X_test, y_train, y_test):
    # TODO: fill this function
    # train by X_train and y_train
    # report the predicting accuracy for both the training and the test data
    pass


def main(argv):
    X_train, X_test, y_train, y_test = gen_train_test_data()

    multinomial_nb(X_train, X_test, y_train, y_test)
    bernoulli_nb(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main(sys.argv)
