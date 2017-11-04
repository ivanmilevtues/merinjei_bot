import numpy as np
from nltk.corpus import stopwords

def add_to_array(array: list, word: str, val: str, features, stemmer):
    if word in stopwords.words():
        return

    word = stemmer.stem(word)
    if word in features:
        array[features.index(word)] += int(val)

def get_unused_dataset_indxs(dataset, threshold=0):
    if threshold == 0:
        threshold = len(dataset) * 0.1
    cols_to_delete = np.sum(dataset, axis=0) < threshold
    indx_to_delete = [indx for indx in range(len(cols_to_delete)) if cols_to_delete[indx]]
    return indx_to_delete