import numpy as np
from nltk.tokenize import RegexpTokenizer
from decorators import not_none
from PreprocessData import PreprocessData


class PreprocessReviewData(PreprocessData):
    """
    PreprocessData -> A class which preprocess a bag of words to a simple
    Numpy vector which can be used for machine learning tasks
    """

    def __init__(self, sub_directories: list, file_names: list, main_dir='data'):
        super().__init__(sub_directories, file_names, main_dir)

    def init_features(self):
        self._open_files()
        pattern = r"([a-z]+|_)"
        tokenizer = RegexpTokenizer(pattern)
        features = set()

        for file in self.files:
            print('tokenizing for ' + file.name + ' started')
            tokens = tokenizer.tokenize(file.read())[:-2]
            print('tokenizing for ' + file.name + ' finished')
            print('features extracting for ' + file.name)
            features.update(self._reduce_tokens(tokens))
            print('features extracted for ' + file.name)

        features.remove('_')

        self._close_files()
        self.features = list(features)
        return features

    @not_none('features')
    def init_dataset(self, pattern=r"([a-z]+.[a-z]+):(\d)"):
        self._open_files()
        dataset = []
        tokenizer = RegexpTokenizer(pattern)
        for file in self.files:
            print("dataset extraction for " + file.name + " started")

            file_lines = file.readlines()
            for line in file_lines:
                label = 1 if 'positive' in line.split('#label#:')[1] else 0
                tokens = tokenizer.tokenize(line)
                dataset.append(self._words_to_array(tokens, label))

            print("dataset extraction for " + file.name + " done")
        self._close_files()
        self.dataset = np.array(dataset)
        return self.dataset

    @staticmethod
    def reduce_dataset(dataset, indx_to_delete):
        return np.delete(dataset, indx_to_delete, 1)

    @staticmethod
    def reduce_features(features, indx_to_delete):
        return [features[i] for i in range(len(features)) if i not in indx_to_delete]


if __name__ == '__main__':
    preprocess = PreprocessReviewData(["books", "dvd", "electronics", "kitchen"], ["negative.review", "positive.review", "unlabeled.review"])
    preprocess.load_features("reduced_full_features.pickle")
    preprocess.init_dataset()
    preprocess.save_dataset("dataset_review_w_reduced_full_features.pickle")
