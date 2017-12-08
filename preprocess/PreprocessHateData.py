import csv
import re
import numpy as np
from collections import Counter
import pickle
from preprocess.decorators import not_none
from preprocess.PreprocessData import PreprocessData
from preprocess.preprocessing_utilities import get_unused_dataset_indxs

class PreprocessHateData(PreprocessData):

    def __init__(self, sub_directories: list, file_names: list,\
                 slang_dict: dict, spell_correct_dict: dict,  main_dir='data'):
        super().__init__(sub_directories, file_names, main_dir)
        self.label_indx = 5
        self.txt_indx = 6
        self.slang_dict = slang_dict
        self.spell_correct_dict = spell_correct_dict

    @not_none('slang_dict')
    @not_none('spell_correct_dict')
    def init_features(self):
        files = self.open_files(self.paths)
        pattern = r"\W+"
        features = set()
        csv_readers = []
        for file in files:
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
        self.close_files(files)
        self.features = list(features)

    @not_none('features')
    def init_dataset(self, pattern=r"\W+"):
        files = self.open_files(self.paths)
        dataset = []
        csv_readers = []
        for file in files:
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

        self.close_files(files)
        self.dataset = np.array(dataset)
        return self.dataset

  
# if __name__ == '__main__':
    # pd = PreprocessHateData([''], ['twitter_hate_speech.csv'], )
    # pd.load_features('reduced_full_features.pickle')
    # pd.init_dataset()

    # pd.save_dataset("dataset_hs_w_reduced_full_features.pickle")