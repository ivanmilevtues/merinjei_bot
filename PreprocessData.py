from abc import abstractmethod
import numpy as np
from nltk.stem import SnowballStemmer
import pickle
from preprocessing_utilities import add_to_array
from decorators import not_none


class PreprocessData:
    """
    PreprocessData -> A class which preprocess a bag of words to a simple
    Numpy vector which can be used for machine learning tasks
    """

    def __init__(self, sub_directories: list, file_names: list, main_dir='main'):
        self.sub_directories = sub_directories
        self.file_names = file_names
        self.main_dir = main_dir
        self.features = None
        self.dataset = None
        self.stemmer = SnowballStemmer("english")
        self.files = []

    @abstractmethod
    def init_features(self):
        pass

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

    @abstractmethod
    @not_none('features')
    def init_dataset(self, pattern=r"([a-z]+.[a-z]+):(\d)"):
       pass

    def load_dataset(self, file='dataset.pickle'):
        with open(file, 'rb') as f:
            self.dataset = pickle.load(f)

    @staticmethod
    def reduce_dataset(dataset, indx_to_delete):
        return np.delete(dataset, indx_to_delete, 1)

    @staticmethod
    def reduce_features(features, indx_to_delete):
        return [features[i] for i in range(len(features)) if i not in indx_to_delete]


    @not_none('dataset')
    def get_dataset(self):
        return self.dataset

    @not_none('dataset')
    def save_dataset(self, file='dataset.pickle'):
        with open(file, 'wb') as f:
            pickle.dump(self.dataset, f)

    @not_none('features')
    def _words_to_array(self, tokens, label):
        result = [0 for _ in self.features]
        for k, v in tokens.items():
            if "_" in k:
                add_to_array(result, k.split('_')[0], v, self.features, self.stemmer)
                add_to_array(result, k.split('_')[1], v, self.features, self.stemmer)
            add_to_array(result, k, v, self.features, self.stemmer)
        result.append(label)
        return result

    def __generate_file_path(self, path):
        return '/'.join(path.split('//'))

    def _open_files(self):
        for file in self.files:
            if not file.closed:
                file.close()

        self.files = []
        for sub_dir in self.sub_directories:
            for file_name in self.file_names:
                path = self.main_dir + "/" + sub_dir + "/" + file_name
                path = self.__generate_file_path(path)
                self.files.append(open(path))

    def _close_files(self):
        for file in self.files:
            file.close()