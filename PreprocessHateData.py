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

                label = 1 if row[self.label_indx] != 2 else 0
                tokens = re.split(pattern, row[self.txt_indx].lower())
                tokens = Counter(word for word in tokens)
                curr_row = self._words_to_array(tokens, label)
                dataset.append(curr_row)
                print(sum(curr_row))

        self.dataset = np.array(dataset)
        return self.dataset


if __name__ == '__main__':
    pd = PreprocessHateData([''], ['twitter_hate_speech.csv'])
    pd.load_features()
    pd.init_dataset()