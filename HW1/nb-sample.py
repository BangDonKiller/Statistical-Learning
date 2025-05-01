#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

SMOOTH_FACTOR = 1


def gen_train_test_data():
    test_ratio = 0.3
    df = pd.read_csv("./HW1/yelp_labelled.txt", sep="\t", header=None, encoding="utf-8")
    count_vect = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    X = count_vect.fit_transform(df[0])
    y = df[1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=0
    )
    return X_train, X_test, y_train, y_test


def multinomial_nb(X_train, X_test, y_train, y_test):
    print("\nMultinomial NB:")

    pos_num = y_train.value_counts()[1]
    neg_num = y_train.value_counts()[0]
    prior_prob = {
        "positive": pos_num / (pos_num + neg_num),
        "negative": neg_num / (pos_num + neg_num),
    }

    conditional_probs = {"positive": {}, "negative": {}}

    for given_y in ["positive", "negative"]:
        repr_y = 1 if given_y == "positive" else 0
        for x_i in range(X_train.shape[1]):
            if SMOOTH_FACTOR:
                conditional_probs[given_y][x_i] = (
                    np.sum(X_train[y_train == repr_y][:, x_i].data) + 1
                ) / (np.sum(X_train[y_train == repr_y].data) + X_train.shape[1] * 1)
            else:
                conditional_probs[given_y][x_i] = np.sum(
                    X_train[y_train == repr_y][:, x_i].data
                ) / np.sum(X_train[y_train == repr_y].data)

    for phase in ["train", "test"]:
        if phase == "train":
            X = X_train
            y = y_train
        else:
            X = X_test
            y = y_test
        predicted = []
        for test_case in X:
            highest_prob = 0
            highest_prob_y = None
            for pos_neg in ["positive", "negative"]:
                prob = prior_prob[pos_neg]
                for x_now in range(test_case.indices.shape[0]):
                    feature = test_case.indices[x_now]
                    times = test_case.data[x_now]
                    prob *= conditional_probs[pos_neg][feature] ** times
                if prob > highest_prob:
                    highest_prob = prob
                    highest_prob_y = pos_neg
            predicted.append(1 if highest_prob_y == "positive" else 0)
        predicted = np.array(predicted)
        print(f"{phase} accuracy: {sum(predicted == y.values) / len(y.values) * 100}%")


def bernoulli_nb(X_train, X_test, y_train, y_test):
    print("\nBernoulli NB:")

    pos_num = y_train.value_counts()[1]
    neg_num = y_train.value_counts()[0]
    prior_prob = {
        "positive": pos_num / (pos_num + neg_num),
        "negative": neg_num / (pos_num + neg_num),
    }

    conditional_probs = {"positive": {}, "negative": {}}

    for given_y in ["positive", "negative"]:
        repr_y = 1 if given_y == "positive" else 0
        for x_i in range(X_train.shape[1]):
            if SMOOTH_FACTOR:
                conditional_probs[given_y][x_i] = (
                    np.count_nonzero(X_train[y_train == repr_y][:, x_i].data) + 1
                ) / (X_train[y_train == repr_y].shape[0] + 2)
            else:
                conditional_probs[given_y][x_i] = (
                    np.count_nonzero(X_train[y_train == repr_y][:, x_i].data)
                    / X_train[y_train == repr_y].shape[0]
                )

    for phase in ["train", "test"]:
        if phase == "train":
            X = X_train
            y = y_train
        else:
            X = X_test
            y = y_test
        predicted = []
        for test_case in X:
            highest_prob = 0
            highest_prob_y = None
            for pos_neg in ["positive", "negative"]:
                prob = prior_prob[pos_neg]
                for x_now in range(test_case.indices.shape[0]):
                    feature = test_case.indices[x_now]
                    prob *= conditional_probs[pos_neg][feature]
                if prob > highest_prob:
                    highest_prob = prob
                    highest_prob_y = pos_neg
            predicted.append(1 if highest_prob_y == "positive" else 0)
        predicted = np.array(predicted)
        print(f"{phase} accuracy: {sum(predicted == y.values) / len(y.values) * 100}%")


def main(argv):
    X_train, X_test, y_train, y_test = gen_train_test_data()

    multinomial_nb(X_train, X_test, y_train, y_test)
    bernoulli_nb(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main(sys.argv)
