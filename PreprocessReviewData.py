import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
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
    def init_dataset(self, pattern=r"([a-z]+.[a-z]+):(\d)"):
        self.__open_files()
        dataset = []
        tokenizer = RegexpTokenizer(pattern)
        for file in self.files:
            print("dataset extraction for " + file.name + " started")

            file_lines = file.readlines()
            for line in file_lines:
                label = 1 if 'positive' in line.split('#label#:')[1] else 0
                tokens = tokenizer.tokenize(line)
                dataset.append(self.__words_to_array(tokens, label))

            print("dataset extraction for " + file.name + " done")

        self.dataset = np.array(dataset)
        return self.dataset

    @staticmethod
    def reduce_dataset(dataset, indx_to_delete):
        return np.delete(dataset, indx_to_delete, 1)

    @staticmethod
    def reduce_features(features, indx_to_delete):
        return [features[i] for i in range(len(features)) if i not in indx_to_delete]

    def __reduce_tokens(self, tokens: list) -> list:
        tokens = [self.stemmer.stem(w) for w in tokens if w not in stopwords.words()]
        return tokens


if __name__ == '__main__':
    preprocess = PreprocessData("", "")
    preprocess.load_dataset()
    labeled_data = preprocess.get_dataset()
    preprocess.load_dataset("unlabled_dataset.pickle")
    unlabeled_data = preprocess.get_dataset()

    dataset = np.concatenate((labeled_data, unlabeled_data), axis=0)
    print(np.count_nonzero(dataset[:, -1:]))
    dataset = PreprocessData.reduce_dataset(dataset)
