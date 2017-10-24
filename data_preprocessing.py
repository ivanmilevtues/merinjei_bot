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

    def __init__(self, sub_directories: list, file_names: list, load_features: bool = False):
        self.sub_directories = sub_directories
        self.file_names = file_names
        self.main_dir = "data"
        self.stemmer = SnowballStemmer("english")
        self.features = None
        if load_features:
            self.load_features()
        self.dataset = None
        self.files = []

    def init_features(self):
        self.__open_files()
        pattern = r"([a-z]+|_)"
        tokenizer = RegexpTokenizer(pattern)
        features = set()

        for file in self.files:
            tokens = tokenizer.tokenize(file.read())[:-2]
            print('tokens are ready')
            features.update(self.__reduce_tokens(tokens))
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
    def init_dataset(self):
        self.__open_files()
        dataset = []
        pattern = r"([a-z]+.[a-z]+):(\d)"
        tokenizer = RegexpTokenizer(pattern)
        for file in self.__open_files():
            file_lines = file.readlines()
            for line in file_lines:
                tokens = tokenizer.tokenize(line)
                dataset.append(self.__words_to_array(tokens, 1 if 'positive' in file.name else 0))

        self.dataset = np.array(dataset)
        return self.dataset

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
        word = self.stemmer(word)
        if word in self.features:
            array[self.features.index(word)] += int(val)

    def __reduce_tokens(self, tokens: list) -> list:
        print(len(tokens))
        print('tokens stemmed')
        tokens = [self.stemmer.stem(w) for w in tokens if w not in stopwords.words()]
        print('done with reduce')
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
    # with open('features.pickle' ) as f:
    #     features = pickle.load(f)

    # print(len(features))
    sub_directories = ["books"]
    data_types = ["positive", "negative"]
    preprocess = PreprocessData(sub_directories, data_types)
    t = time.time()
    features = preprocess.init_features()
    print(t - time.time())
    print(len(features))
    preprocess.save_features()