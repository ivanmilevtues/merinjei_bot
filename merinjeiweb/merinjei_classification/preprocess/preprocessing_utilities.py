import time
import pickle
import numpy as np
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score, recall_score, f1_score,\
                            classification_report


def add_to_array(array: list, word: str, val: str, features, stemmer):
    if word in stopwords.words():
        return

    word = stemmer.stem(word)
    if word in features:
        array[features.index(word)] += int(val)


def get_unused_dataset_indxs(dataset, bottom_threshold, top_threshold):
    summed_dataset = np.sum(dataset, axis=0)
    bottom_indxs_delete = summed_dataset <= bottom_threshold
    top_indxs_delete = summed_dataset >= top_threshold
    cols_to_delete = np.logical_or(bottom_indxs_delete, top_indxs_delete)

    return [indx for indx in range(len(cols_to_delete)) if cols_to_delete[indx]]


def get_unused_features(features_importance, threshold=0):
    cols_to_delete = features_importance <= threshold
    return [indx for indx in range(len(cols_to_delete)) if cols_to_delete[indx]]


def concat_features(features_a, features_b, file):
    features_a = set(features_a)
    features_a.update(features_b)
    features_a = list(features_a)
    with open(file, 'wb') as f:
        pickle.dump(features_a, f)


def split_to_train_test(features_and_labels: list, test_set_percent=0.4,
                        shuffle=True, labels=None):
    features = features_and_labels

    if labels is None:
        labels = features_and_labels[:, -1:].ravel()
        features = features_and_labels[:, :-1]
    else:
        labels = labels.ravel()

    return train_test_split(features, labels, test_size=test_set_percent,
                            random_state=42, shuffle=shuffle, stratify=labels)
