import csv
import re
import numpy as np
from collections import Counter
import pickle
from preprocess.decorators import not_none
from preprocess.PreprocessData import PreprocessData
from preprocess.AutoCorrect import AutoCorrect
from sklearn.feature_extraction.text import TfidfVectorizer


class PreprocessHateData(PreprocessData):

    def __init__(self, sub_directories: list, file_names: list,\
                 slang_dict: dict, spell_correct_dict: dict,  main_dir='data', ngrams=3):
        super().__init__(sub_directories, file_names, main_dir)
        self.label_indx = 5
        self.txt_indx = 6
        self.slang_dict = slang_dict
        self.spell_correct_dict = spell_correct_dict
        self.ngram_vectorizer = TfidfVectorizer(min_df=5,
                                                max_df=0.501,
                                                max_features=50000,
                                                ngram_range=(1, ngrams),
                                                token_pattern=r'\b\w+\b')

    @not_none('slang_dict')
    @not_none('spell_correct_dict')
    def init_dataset(self, pattern=r"\W+"):
        files = self.open_files(self.paths)
        dataset = []
        csv_readers = []
        labels = []

        for file in files:
            csv_readers.append(csv.reader(file))

        for reader in csv_readers:
            skip_first = True
            for row in reader:
                if skip_first:
                    skip_first = False
                    continue

                label = 1 if int(row[self.label_indx]) == 2 else 0
                labels.append(label)

                tweet = row[self.txt_indx].lower()
                tweet = self.__replace_mentions_urls(tweet)
                tokens = re.split(pattern, tweet)
                tokens = self.__spell_check(tokens)
                tokens = self._reduce_tokens(tokens)
                curr_row = ' '.join(tokens)

                # Adding the reduced sentence to the dataset(corpus)
                dataset.append(curr_row)

        self.close_files(files)

        # Changes the array to be vector
        labels = np.array(labels).reshape((len(labels), 1))

        self.dataset = self.ngram_vectorizer.fit_transform(dataset).toarray()
        return (self.dataset, labels)

    def get_analizer(self):
        return self.ngram_vectorizer.build_analyzer()
    
    def save_ngram_vectorizer(self, file='vectorizer.pkl'):
        with open(file, 'wb') as f:
            pickle.dump(self.ngram_vectorizer, f)

    def __replace_mentions_urls(self, tweet):
        url_regex = re.compile(
            r'(?:http|ftp)s?://'
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
            r'localhost|'
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
            r'(?::\d+)?' 
            r'(?:/?|[/?]\S+)', re.IGNORECASE)

        mentions_regex = re.compile(r'@\w+')
        hashtag_regex = re.compile(r'#\w+')

        tweet = 'URL'.join(re.split(url_regex, tweet))
        tweet = 'MENTION'.join(re.split(mentions_regex, tweet))
        tweet = 'HASHTAG'.join(re.split(hashtag_regex, tweet))

        return tweet

    def __spell_check(self, tokens):
        res = []
        for token in tokens:
            if token in self.slang_dict:
                res.append(self.slang_dict[token])
            elif token in self.spell_correct_dict:
                res.append(self.spell_correct_dict[token])
            else:
                res.append(token)
        return res
