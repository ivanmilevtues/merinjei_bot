import re
import nltk
import pickle
import numpy as np
from collections import Counter
from merinjei_classification.preprocess.PreprocessData import PreprocessData
from merinjei_classification.preprocess.decorators import not_none

class PreprocessQuestions(PreprocessData):

    def __init__(self, sub_dirs: list, file_names: list,
                 main_dir='merinjei_classification/data'):
        super().__init__(sub_dirs, file_names, main_dir )
        self.labels = ['ABBR', 'DESC', 'PROCEDURE', 'HUM', 'LOC', 'NUM']

    def init_features(self):
        files = self.open_files(self.paths)
        pattern = r'([^a-zA-Z0-9_\'])+'
        features = set()

        for file in files:
            lines = file.readlines()
            for line in lines:
                tokens = re.split(pattern, line)[1:] # we take everything without the label
                tokens = list(filter(None, tokens))
                tags = [pos for _, pos in nltk.pos_tag(tokens)]
                features.update(self._reduce_tokens(tokens))
                features.update(tags)
        self.close_files(files)
        self.features = list(features)
        return self.features

    @not_none('features')
    def init_dataset(self):
        files = self.open_files(self.paths)
        pattern = r'([^a-zA-Z0-9_\'])+'
        dataset = []
        for file in files:
            lines = file.readlines()
            for line in lines:
                tokens = re.split(pattern, line)
                label, tokens = tokens[0], tokens[1:]
                label = self._pick_label(label)
                if label is None:
                    continue
                tokens = list(filter(None, tokens))
                print(tokens)
                pos_tags = [pos for _, pos in nltk.pos_tag(tokens)]
                tokens += pos_tags
                tokens = Counter(word for word in tokens)
                dataset.append(self._words_to_array(tokens, label))

        self.close_files(files)
        self.dataset = np.array(dataset)
        return self.dataset

    @not_none('labels')
    def save_labels(self, file="data/processed_data/question_labels.pickle"):
        with open('file', 'wb') as f:
            pickle.dump(self.labels, f)

    def _pick_label(self, label):
        for indx in range(len(self.labels)):
            if self.labels[indx] in label:
                return indx
