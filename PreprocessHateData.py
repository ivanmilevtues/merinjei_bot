import csv
import re
import numpy as np
from collections import Counter
from decorators import not_none

from PreprocessData import PreprocessData

# TODO
# Add autocorrection!

class PreprocessHateData(PreprocessData):

    def __init__(self, sub_directories: list, file_names: list, main_dir='data'):
        super().__init__(sub_directories, file_names, main_dir)
        self.label_indx = 5
        self.txt_indx = 6

    def init_features(self):
        pass

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


if __name__ == '__main__':
    pd = PreprocessHateData([''], ['twitter_hate_speech.csv'])
    pd.load_dataset("hatespeech_dataset.pickle")
    pd.balance_dataset()
    ds = pd.get_dataset()
    pd.load_features()
    print("FEATUES")
    print(len(pd.get_features()))
    print(ds.shape)
    print(sum(ds[:, -1:]))
    print(len(ds))
    print(ds)