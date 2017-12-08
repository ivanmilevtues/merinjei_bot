from abc import abstractmethod
import numpy as np
from nltk.stem import SnowballStemmer
import pickle
from nltk.corpus import stopwords
from preprocess.preprocessing_utilities import add_to_array, concat_features
from preprocess.decorators import not_none
from preprocess.FileOpenerMixin import FileOpenerMixin


class PreprocessData(FileOpenerMixin):
    """
    PreprocessData -> A class which preprocess a bag of words to a simple
    Numpy vector which can be used for machine learning tasks
    """

    def __init__(self, sub_directories: list, file_names: list, main_dir='main'):
        self.paths = {'main_dir': main_dir, 'sub_directories': sub_directories, 'file_names': file_names}
        self.features = None
        self.dataset = None
        self.stemmer = SnowballStemmer("english")

    @not_none('dataset')
    def balance_dataset(self):
        labels = self.dataset[:, -1:]
        labels_sum = np.sum(labels)
        balance = labels_sum if labels_sum < len(labels) // 2 else len(labels) - labels_sum
        # sort the dataset by its labels
        self.dataset = self.dataset[self.dataset[:, -1].argsort()]

        negatives = self.dataset[0: balance,: ]
        postives = self.dataset[-balance: -1, :]

        self.dataset = np.concatenate((negatives, postives), axis=0)

        return self.dataset

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
    
    def load_and_get_features(self, file="features.pickle"):
        with open(file, "rb") as f:
            self.features = pickle.load(f)
        return self.features

    @abstractmethod
    @not_none('features')
    def init_dataset(self, pattern=r"([a-z]+.[a-z]+):(\d)"):
       pass

    def load_dataset(self, file='dataset.pickle'):
        with open(file, 'rb') as f:
            self.dataset = pickle.load(f)
    
    def load_and_get_dataset(self, file='dataset.pickle'):
        with open(file, 'rb') as f:
            self.dataset = pickle.load(f)
        return self.dataset

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
        tokens = tokens if type(tokens) is list else tokens.items()
        for k, v in tokens:
            if "_" in k:
                add_to_array(result, k.split('_')[0], v, self.features, self.stemmer)
                add_to_array(result, k.split('_')[1], v, self.features, self.stemmer)
            add_to_array(result, k, v, self.features, self.stemmer)
        result.append(label)
        return result

    @not_none('dataset')
    def add_len_feature(self):
        dataset = self.dataset[:, :-1]
        labels = self.dataset[:, -1:]

        dataset_lens = np.sum(dataset, axis=1)
        dataset_lens = np.array([dataset_lens])

        dataset = np.append(dataset.T, dataset_lens, axis=0).T
        self.dataset = np.append(dataset, labels, axis=1)

    def _reduce_tokens(self, tokens: list) -> list:
        tokens = [self.stemmer.stem(w) for w in tokens if w not in stopwords.words()]
        return tokens



if __name__ == "__main__":
    pd = PreprocessData([], [])
    pd.load_features()
    all_review_features = pd.get_features()
    pd.load_features("hs_features.pickle")
    all_hs_features = pd.get_features()
    print(len(all_review_features), len(all_hs_features))

    pd.load_features("reduced_features.pickle")
    reduced_review_features = pd.get_features()
    pd.load_features("reduced_hs_features.pickle")
    reduced_hs_features = pd.get_features()
    print(len(reduced_review_features), len(reduced_hs_features))

    concat_features(reduced_hs_features, reduced_review_features, "reduced_full_features.pickle")
    concat_features(all_review_features, all_hs_features, "full_features.pickle")