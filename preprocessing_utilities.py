import numpy as np
from nltk.corpus import stopwords

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

    cols_to_delete = np.logical_and(bottom_indxs_delete, top_indxs_delete)
    indx_to_delete = [indx for indx in range(len(cols_to_delete)) if cols_to_delete[indx]]

    return indx_to_delete


def get_unused_features(features_importance, threshold=0):
    cols_to_delete = features_importance <= threshold
    indx_to_delete = [indx for indx in range(len(cols_to_delete)) if cols_to_delete[indx]]
    return indx_to_delete