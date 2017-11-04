import numpy as np
from nltk import word_tokenize
from preprocessing_utilities import add_to_array
from nltk.stem import SnowballStemmer


class LineParser():

    def __init__(self, features):
        self.features = features
        self.stemmer = SnowballStemmer('english')

    def parse_line(self, line):
        line = line[:-1].lower()
        tokens = word_tokenize(line)
        mapped_tokens = self.__tokens_to_map(tokens)
        dataset = self.__map_to_dataset(mapped_tokens)
        return np.array(dataset)

    def __tokens_to_map(self, tokens):
        res = {}
        for token in tokens:
            if token not in res.items():
                res[token] = 1
            else:
                res[token] += 1
        return res

    def __map_to_dataset(self, data):
        result = [0 for _ in self.features]
        for k, v in data.items():
            add_to_array(result, k, v, self.features, self.stemmer)
        return result
