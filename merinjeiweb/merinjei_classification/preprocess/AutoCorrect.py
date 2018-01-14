import pickle

from merinjei_classification.preprocess.FileOpenerMixin import FileOpenerMixin
from merinjei_classification.preprocess.decorators import not_none


# TODO add to classes one for the slang and one for the spell correction
# or add path changing method


class AutoCorrect(FileOpenerMixin):
    def __init__(self, sub_dirs: list, files: list, main_dir='data'):
        self.paths = {'main_dir': main_dir, 'sub_directories': sub_dirs, 'file_names': files}
        self.slang_dict = None
        self.spell_correct_dict = None
        
    def init_slang_dict(self):
        files = self.open_files(self.paths)
        slang_dict = {}
        content = []
        for file in files:
            content += file.readlines()

        for line in content:
            line = line.strip().lower()
            if '`' in line:
                k, v = line.split('`')
                slang_dict[k] = v
        self.slang_dict = slang_dict
        self.close_files(files)
        return self.slang_dict

    @not_none('slang_dict')
    def save_slang_dict(self, file='slang_correction.pkl'):
        with open(file, 'wb') as f:
            pickle.dump(self.slang_dict, f)

    def load_slang_dict(self, file='slang_correction.pkl'):
        with open(file, 'rb') as f:
            self.slang_dict = pickle.load(f)
    
    def load_and_get_slang_dict(self,  file='slang_correction.pkl'):
        with open(file, 'rb') as f:
            self.slang_dict = pickle.load(f)
        return self.slang_dict

    def init_spell_correction(self):
        files = self.open_files(self.paths)
        spell_correct_dict = {}
        content = []
        for file in files:
            content += file.readlines()

        for line in content:
            line = line.strip()
            v, k = line.split(': ')
            for key in k.split():
                spell_correct_dict[key] = v

        self.close_files(files)
        self.spell_correct_dict = spell_correct_dict
        return self.spell_correct_dict

    @not_none('spell_correct_dict')
    def save_spell_correction(self, file="spell_correct.pkl"):
        with open(file, 'wb') as f:
            pickle.dump(self.spell_correct_dict, f)

    def load_spell_correct(self, file="spell_correct.pkl"):
        with open(file, 'rb') as f:
            self.spell_correct_dict = pickle.load(f)

    def load_and_get_spell_correct(self, file="spell_correct.pkl"):
        with open(file, 'rb') as f:
            self.spell_correct_dict = pickle.load(f)
        return self.spell_correct_dict