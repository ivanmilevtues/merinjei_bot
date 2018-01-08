import re
import pickle
import numpy as np
from collections import Counter
from preprocess.PreprocessData import PreprocessData
from preprocess.decorators import not_none
import nltk


class PreprocessCocoQuestions(PreprocessData):

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
                tokens = re.split(pattern, line)
                tags = [pos for _, pos in nltk.pos_tag(tokens)]
                features.update(self._reduce_tokens(tokens))
                features.update(tags)
        self.close_files(files)
        self.features = list(features)
        return self.features

    @not_none('features')
    def init_dataset(self, label_file='data/questions/types.txt'):
        files = self.open_files(self.paths)

        with open(label_file, 'r') as f:
            labels = list(map(int, f.readlines()))

        pattern = r'\s'
        dataset = []
        for file in files:
            lines = file.readlines()
            for indx in range(len(lines)):
                tokens = lines[indx].split()
                tags = nltk.pos_tag(tokens)
                tokens += tags
                tokens = Counter(word for word in tokens)
                dataset.append(self._words_to_array(tokens, labels[indx]))
        self.close_files(files)
        self.dataset = np.array(dataset)
        return self.dataset
