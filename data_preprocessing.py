import time
import numpy as np
from decorators import not_none
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import pickle

class PreprocessData:
    """
    PreprocessData -> A class which preprocess a bag of words to a simple
    Numpy vector which can be used for machine learning tasks
    """

    def __init__(self, sub_directories: list, file_names: list):
        self.sub_directories = sub_directories
        self.file_names = file_names
        self.main_dir = "data"
        self.stemmer = SnowballStemmer("english")
        self.features = None
        self.dataset = None
        self.files = []

    def init_features(self):
        self.__open_files()
        pattern = r"([a-z]+|_)"
        tokenizer = RegexpTokenizer(pattern)
        features = set()

        for file in self.files:
            print('tokenizing for ' + file.name + ' started')
            tokens = tokenizer.tokenize(file.read())[:-2]
            print('tokenizing for ' + file.name + ' finished')
            print('features extracting for ' + file.name)
            features.update(self.__reduce_tokens(tokens))
            print('features extracted for ' + file.name)

        features.remove('_')

        self.__close_files()
        self.features = list(features)
        return features

    @not_none('features')
    def get_features(self):
        return self.features

    @not_none('features')
    def save_features(self, file="features.pickle"):
        with open(file, "wb") as f:
            pickle.dump(self.features, f)

    def load_features(self, file="features.pickle"):
        with open(file, "rb") as f:
            self.features = pickle.load(f)

    @not_none('features')
    def init_dataset(self, pattern=r"([a-z]+.[a-z]+):(\d)"):
        self.__open_files()
        dataset = []
        tokenizer = RegexpTokenizer(pattern)
        for file in self.files:
            label = 1 if 'positive' in file.name else 0

            print("dataset extraction for " + file.name + " started")
            file_lines = file.readlines()
            for line in file_lines:
                if 'unlabeled' in file.name:
                    label = 1 if 'positive' in line.split('#label#:')[1] else 0
                tokens = tokenizer.tokenize(line)
                dataset.append(self.__words_to_array(tokens, label))
            print("dataset extraction for " + file.name + " done")

        self.dataset = np.array(dataset)
        return self.dataset

    def load_dataset(self, file='dataset.pickle'):
        with open(file, 'rb') as f:
            self.dataset = pickle.load(f)

    @not_none('dataset')
    def get_dataset(self):
        return self.dataset

    @not_none('dataset')
    def save_dataset(self, file='dataset.pickle'):
        with open(file, 'wb') as f:
            pickle.dump(self.dataset, f)


    @not_none('features')
    def __words_to_array(self, tokens, label):
        result = [0 for _ in self.features]
        for k, v in tokens:
            if "_" in k:
                self.__add_to_array(result, k.split('_')[0], v)
                self.__add_to_array(result, k.split('_')[1], v)
            self.__add_to_array(result, k, v)
        result.append(label)
        return result


    @not_none('features')
    def __add_to_array(self, array: list, word: str, val: str):
        if word in stopwords.words():
            return
        word = self.stemmer.stem(word)
        if word in self.features:
            array[self.features.index(word)] += int(val)

    def __reduce_tokens(self, tokens: list) -> list:
        tokens = [self.stemmer.stem(w) for w in tokens if w not in stopwords.words()]
        return tokens

    def __open_files(self):
        for file in self.files:
            if not file.closed:
                file.close()

        self.files = []
        for sub_dir in self.sub_directories:
            for file_name in self.file_names:
                self.files.append(open(self.main_dir + "/" + sub_dir + "/" + file_name + ".review"))

    def __close_files(self):
        for file in self.files:
            file.close()


if __name__ == '__main__':

    sub_directories = ["books", "dvd", "electronics", "kitchen"]
    data_types = ['unlabeled']
    preprocess = PreprocessData(sub_directories, data_types)
    preprocess.load_features()
    preprocess.init_dataset()
    preprocess.save_dataset(file='unlabled_dataset.pickle')
