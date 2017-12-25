import re
import csv
import nltk
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
        self.ngram_vectorizer = TfidfVectorizer(use_idf=True,
                                                min_df=5,
                                                max_df=0.501,
                                                max_features=10000,
                                                ngram_range=(1, ngrams),
                                                token_pattern=r'\b\w+\b')

    @not_none('slang_dict')
    @not_none('spell_correct_dict')
    def init_dataset(self, pattern=r"\W+"):
        pos_data = self.init_pos_tags_ds()
        ngram_data, labels = self.init_ngrams_datset()
        self.dataset = np.concatenate([pos_data, ngram_data], axis=1)
        return (self.dataset, labels)

    def init_pos_tags_ds(self, pattern=r"\W+"):
        files = self.open_files(self.paths)
        dataset = []
        csv_readers = []
        labels = []
        pos_vectorizer = TfidfVectorizer(
            use_idf=False,
            ngram_range=(1, 3),
            min_df=5,
            max_df=0.75,
            max_features=5000
        )

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
                tokens = nltk.word_tokenize(tweet)
                tokens_tagged = nltk.pos_tag(tokens)
                # Taking only the part of speech (Word, PartOfSpeech)
                pos_tags = [pos[1] for pos in tokens_tagged]
                curr_row = ' '.join(pos_tags)

                dataset.append(curr_row)
        return pos_vectorizer.fit_transform(dataset).toarray()


    def init_ngrams_datset(self, pattern=r"\W+"):
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
                # tokens = self.__spell_check(tokens) # remove it for faster parsking
                tokens = self._reduce_tokens(tokens)
                curr_row = ' '.join(tokens)

                # Adding the reduced sentence to the dataset(corpus)
                dataset.append(curr_row)

        self.close_files(files)

        # Changes the array to be vector
        labels = np.array(labels).reshape((len(labels), 1))

        dataset = self.ngram_vectorizer.fit_transform(dataset).toarray()
        return (dataset, labels)

    def get_analizer(self):
        return self.ngram_vectorizer.build_analyzer()
    
    def save_ngram_vectorizer(self, file='vectorizer.pkl'):
        with open(file, 'wb') as f:
            pickle.dump(self.ngram_vectorizer, f)

    def __replace_mentions_urls(self, tweet, replace_url='', replace_mention='', replace_hashtag=''):
        url_regex = re.compile(
            r'(?:http|ftp)s?://'
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
            r'localhost|'
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
            r'(?::\d+)?' 
            r'(?:/?|[/?]\S+)', re.IGNORECASE)

        mentions_regex = re.compile(r'@\w+')
        hashtag_regex = re.compile(r'#\w+')

        tweet = replace_url.join(re.split(url_regex, tweet))
        tweet = replace_mention.join(re.split(mentions_regex, tweet))
        tweet = replace_hashtag.join(re.split(hashtag_regex, tweet))

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
