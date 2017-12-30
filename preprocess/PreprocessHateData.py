import re
import csv
import nltk
from nltk.sentiment.util import demo_vader_instance
import numpy as np
from collections import Counter
import pickle
from preprocess.decorators import not_none
from preprocess.PreprocessData import PreprocessData
from preprocess.AutoCorrect import AutoCorrect
from sklearn.feature_extraction.text import TfidfVectorizer


class PreprocessHateData(PreprocessData):

    def __init__(self, sub_directories: list, file_names: list, main_dir='data'):
        super().__init__(sub_directories, file_names, main_dir)
        self.label_indx = 5
        self.txt_indx = 6
        self.corpus = None

    def generate_corpus_get_labels(self):
        files = self.open_files(self.paths)
        csv_readers = []
        self.corpus = []
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
                self.corpus.append(tweet)

        self.close_files(files)
        return np.array(labels)

    def init_dataset(self, pattern=r"\W+"):
        labels = self.generate_corpus_get_labels()
        ngram_data = self.init_ngrams_datset()
        pos_data = self.init_pos_tags_ds()
        additional_features = self.init_other_features()
        self.dataset = np.concatenate([ngram_data, pos_data, additional_features], axis=1)
        return (self.dataset, labels)

    @not_none('corpus')
    def init_pos_tags_ds(self, pattern=r"\W+"):
        dataset = []
        pos_vectorizer = TfidfVectorizer(
            use_idf=False,
            ngram_range=(1, 3),
            min_df=5,
            max_df=0.75,
            max_features=5000
        )

        for tweet in self.corpus:
            tweet = self.__replace_mentions_urls(tweet)
            tokens = nltk.word_tokenize(tweet)
            tokens_tagged = nltk.pos_tag(tokens)
            # Taking only the part of speech (Word, PartOfSpeech)
            pos_tags = [pos[1] for pos in tokens_tagged]
            curr_row = ' '.join(pos_tags)
            dataset.append(curr_row)

        return pos_vectorizer.fit_transform(dataset).toarray()
  
    @not_none('corpus')
    def init_other_features(self):
        dataset = []
        for tweet in self.corpus:
            tweet = self.__replace_mentions_urls(tweet, "URL", "MENTION", "HASHTAG")
            
            url_count = tweet.count("URL")
            mention_count = tweet.count('MENTION')
            hashtag_count = tweet.count("HASHTAG")

            number_of_words = re.split(r'[a-z]+', tweet)
            number_of_terms = len(tweet.split())
            number_of_unique_terms = len(set(tweet.split()))
            number_of_syllables = len(re.split(r'[aeiouy]', tweet))
            number_of_chars = len(re.split(r'[a-z]', tweet))
            number_of_chars_total = len(re.split(r'[a-z]|\W', tweet))

            avrg_syllables = number_of_syllables / number_of_words
            # Flesch–Kincaid grade level
            fkra_score = (0.39 * number_of_words) + ( 11.8 * number_of_syllables / number_of_words) - 15.59

            # Flesch reading ease
            fre_score = 206.835 - (1.015 * number_of_words) - (84.6 * number_of_syllables / number_of_words)

            tweet.replace('URL', '')
            tweet.replace('MENTION', '')
            tweet.replace('HASHTAG', '')
            polarity = demo_vader_instance(tweet)

            dataset.append([url_count, mention_count, hashtag_count, number_of_words,\
                            number_of_terms, number_of_unique_terms,number_of_syllables,\
                            number_of_chars, number_of_chars_total, avrg_syllables,\
                            fkra_score, fre_score,\
                            polarity['neg'], polarity['neu'], polarity['pos']])

        return np.array(dataset)

    @not_none('corpus')
    def init_ngrams_datset(self, pattern=r"\W+"):
        files = self.open_files(self.paths)
        dataset = []

        ngram_vectorizer = TfidfVectorizer(use_idf=True,
                                           min_df=5,
                                           max_df=0.501,
                                           max_features=10000,
                                           ngram_range=(1, 3),
                                           token_pattern=r'\b\w+\b')

        for tweet in self.corpus:
            tweet = self.__replace_mentions_urls(tweet)
            tokens = re.split(pattern, tweet)
            tokens = self._reduce_tokens(tokens)
            curr_row = ' '.join(tokens)

            # Adding the reduced sentence to the dataset(corpus)
            dataset.append(curr_row)

        self.close_files(files)
        return ngram_vectorizer.fit_transform(dataset).toarray()
 
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
