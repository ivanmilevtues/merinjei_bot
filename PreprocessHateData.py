import csv
import re
import numpy as np
from collections import Counter
from decorators import not_none
import pickle
from PreprocessData import PreprocessData
from preprocessing_utilities import get_unused_dataset_indxs

class PreprocessHateData(PreprocessData):

    def __init__(self, sub_directories: list, file_names: list, main_dir='data'):
        super().__init__(sub_directories, file_names, main_dir)
        self.label_indx = 5
        self.txt_indx = 6
        self.slang_dict = None
        self.spell_correct_dict = None

    @not_none('slang_dict')
    @not_none('spell_correct_dict')
    def init_features(self):
        self._open_files()
        pattern = r"\W+"
        features = set()
        csv_readers = []
        for file in self.files:
            csv_readers.append(csv.reader(file))

        for reader in csv_readers:
            skip_first = True
            for row in reader:
                if skip_first:
                    skip_first = False
                    continue
                tokens = re.split(pattern, row[self.txt_indx].lower())
                for indx in range(len(tokens)):
                    word = tokens[indx]
                    if word in self.slang_dict:
                        tokens += self.slang_dict[word].split()
                        del tokens[indx]
                    if word in self.spell_correct_dict:
                        tokens[indx] = self.spell_correct_dict[word]

                features.update(self._reduce_tokens(tokens))
        self.features = list(features)

    @not_none('dataset')
    def balance_dataset(self):
        labels = self.dataset[:, -1:]
        labels_sum = np.sum(labels)
        balance = labels_sum if labels_sum < len(labels) // 2 else len(labels) - labels_sum

        # sort the dataset by its labels
        self.dataset[self.dataset[:, -1:].argsort()]

        negatives = self.dataset[0: balance,: ]
        postives = self.dataset[-balance: -1, :]

        self.dataset = np.concatenate((negatives, postives), axis=0)

        return self.dataset

    @not_none('features')
    def init_dataset(self, pattern=r"\W+"):
        self._open_files()
        dataset = []
        csv_readers = []
        for file in self.files:
            csv_readers.append(csv.reader(file))

        for reader in csv_readers:
            skip_first = True
            for row in reader:
                if skip_first:
                    skip_first = False
                    continue

                label = 1 if int(row[self.label_indx]) == 2 else 0
                tokens = re.split(pattern, row[self.txt_indx].lower())
                tokens = Counter(word for word in tokens)
                curr_row = self._words_to_array(tokens, label)
                dataset.append(curr_row)

        self._close_files()
        self.dataset = np.array(dataset)
        return self.dataset

    def init_slang_dict(self, main_dir, sub_dirs, files):
        self._open_files({'main_dir': main_dir, 'sub_directories': sub_dirs, 'file_names': files})
        slang_dict = {}
        content = []
        for file in self.files:
            content += file.readlines()

        for line in content:
            line = line.strip().lower()
            if '`' in line:
                k, v = line.split('`')
                slang_dict[k] = v
        self.slang_dict = slang_dict

    @not_none('slang_dict')
    def save_slang_dict(self, file='slang_correction.pickle'):
        with open(file, 'wb') as f:
            pickle.dump(self.slang_dict, f)

    def load_slang_dict(self, file='slang_correction.pickle'):
        with open(file, 'rb') as f:
            self.slang_dict = pickle.load(f)

    def init_spell_correction(self, main_dir, sub_dirs, files):
        self._open_files({'main_dir': main_dir, 'sub_directories': sub_dirs, 'file_names': files})
        spell_correct_dict = {}
        content = []
        for file in self.files:
            content += file.readlines()

        for line in content:
            line = line.strip()
            print(line.split(": "))
            v, k = line.split(': ')
            for key in k.split():
                spell_correct_dict[key] = v

        self._close_files()
        self.spell_correct_dict = spell_correct_dict

    @not_none('spell_correct_dict')
    def save_spell_correction(self, file="spell_correct.pickle"):
        with open(file, 'wb') as f:
            pickle.dump(self.spell_correct_dict, f)

    def load_spell_correct(self, file="spell_correct.pickle"):
        with open(file, 'rb') as f:
            self.spell_correct_dict = pickle.load(f)

if __name__ == '__main__':
    pd = PreprocessHateData([''], ['twitter_hate_speech.csv'])
    pd.load_dataset('hs_dataset.pickle')
    pd.load_features('hs_features.pickle')
    print(len(pd.get_features()))
    indexes = get_unused_dataset_indxs(pd.get_dataset(), 2, 20000)
    print(len(indexes))
    ds = PreprocessData.reduce_dataset(pd.get_dataset(), indexes)
    features = PreprocessData.reduce_features(pd.get_features(),  indexes)
    print(len(features))
    pd.features = features
    pd.dataset = ds
    pd.save_features('reduced_hs_features.pickle')
    pd.save_dataset('reduced_hs_dataset.pickle')