import re
import json
import numpy as np
from preprocess.PreprocessQuestions import PreprocessQuestions
from preprocess.decorators import not_none
from collections import Counter

class PreprocessJSONQuestions(PreprocessQuestions):

    def __init__(self, sub_dirs: list, file_names: list, main_dir='data'):
        super().__init__(sub_dirs, file_names, main_dir)
        self.labels = ['ABBR', 'DESC', 'PROCEDURE', 'PERSON', 'LOCATION', 'NUMBER', 'ORGANIZATION', 'CAUSALITY']

    def init_features(self):
        files = self.open_files(self.paths)
        pattern = r'([^a-zA-Z0-9_'])+'
        features = set()
        data_dicts = []

        for f in files:
            data_dicts.append(json.loads(f.read()))
        
        for data_dict in data_dicts:
            for entry in data_dict:
                tokens = re.split(pattern, entry['q_en'])
                tokens = Counter(word for word in tokens)
                features.update(self._reduce_tokens(tokens))

        self.close_files(files)
        self.features = list(features)
        return self.features

    @not_none('features')
    def init_dataset(self):
        files = self.open_files(self.paths)
        # all non character chars
        pattern = r'([^a-zA-Z0-9_'])+'
        dataset = []
        data_dicts = []

        for f in files:
            data_dicts.append(json.loads(f.read()))
        
        for data_dict in data_dicts:
            for entry in data_dict:
                if entry['q_en'] is '':
                    continue

                label = self._pick_label(entry['q_type'])
                if label is None:
                    continue

                tokens = re.split(pattern, entry['q_en'])
                tokens = Counter(word for word in tokens)
                dataset.append(self._words_to_array(tokens, label))

        self.close_files(files)
        self.dataset = np.array(dataset)
        return self.dataset