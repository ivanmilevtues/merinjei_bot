import numpy as np
from nltk import word_tokenize
from preprocessing_utilities import add_to_array
from collections import Counter
from nltk.stem import SnowballStemmer


class LineParser():

    def __init__(self, features):
        self.features = features
        self.stemmer = SnowballStemmer('english')

    def parse_line(self, line):
        line = line[:-1].lower()
        mapped_dataset = Counter(w for w in line.split(' '))
        dataset = self.__map_to_dataset(mapped_dataset)
        return np.array(dataset)

    def __map_to_dataset(self, data):
        result = [0 for _ in self.features]
        for k, v in data.items():
            add_to_array(result, k, v, self.features, self.stemmer)
        return result