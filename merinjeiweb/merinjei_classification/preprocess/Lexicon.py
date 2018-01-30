import csv
from merinjei_classification.preprocess.decorators import not_none
from merinjei_classification.preprocess.FileOpenerMixin import FileOpenerMixin
from collections import OrderedDict


class Lexicon(FileOpenerMixin):
    def __init__(self, sub_directories: list, file_names: list, main_dir='data'):
        self.paths = {
            'main_dir': main_dir, 'sub_directories': sub_directories, 'file_names': file_names}
        self.lexicon = None

    def init_lexicon(self):
        self.lexicon = OrderedDict()

        files = self.open_files(self.paths)
        csv_readers = []
        for file in files:
            csv_readers.append(csv.reader(file))

        for reader in csv_readers:
            skip_first = True
            for row in reader:
                if skip_first:
                    skip_first = False
                    continue
                self.lexicon[row[0]] = float(row[1])

        self.close_files(files)

    @not_none('lexicon')
    def get_lexicon(self):
        return self.lexicon

