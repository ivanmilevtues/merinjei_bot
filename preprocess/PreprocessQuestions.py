import re
import numpy as np
from collections import Counter
from preprocess.PreprocessData import PreprocessData
from preprocess.decorators import not_none

class PreprocessQuestions(PreprocessData):

    def __init__(self, sub_dirs: list, file_names: list, main_dir='data'):
        super().__init__(sub_dirs, file_names, main_dir )
        self.labels = []
    

    def init_features(self):
        files = self.open_files(self.paths)
        pattern = r"\s"
        features = set()

        for file in files:
            lines = file.readlines()
            for line in lines:
                tokens = re.split(pattern, line)[1:] # we take everything without the label
                features.update(self._reduce_tokens(tokens))
        self.close_files(files)
        self.features = list(features)
        return self.features

    @not_none('features')
    def init_dataset(self):
        files = self.open_files(self.paths)
        pattern = r'\s'
        dataset = []
        for file in files:
            lines = file.readlines()
            for line in lines:
                tokens = re.split(pattern, line)
                label, tokens = tokens[0], tokens[1:]
                tokens = Counter(word for word in tokens)
                label = self.__pick_label(label)
                dataset.append(self._words_to_array(tokens, label))

        self.close_files(files)
        self.dataset = np.array(dataset)
        return self.dataset

    def __pick_label(self, label):
        if label not in self.labels:
            self.labels.append(label)
        return self.labels.index(label)