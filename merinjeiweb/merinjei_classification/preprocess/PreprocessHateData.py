import re
import csv
import nltk
import pickle
import numpy as np
from collections import Counter
from nltk.util import trigrams, bigrams
from nltk.stem import SnowballStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from merinjei_classification.preprocess.decorators import not_none
from merinjei_classification.preprocess.PreprocessData import PreprocessData
from merinjei_classification.preprocess.AutoCorrect import AutoCorrect
from merinjei_classification.preprocess.Lexicon import Lexicon


class PreprocessHateData(PreprocessData):

    def __init__(self, sub_directories: list, file_names: list, main_dir='data'):
        super().__init__(sub_directories, file_names, main_dir)
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.label_indx = 5
        self.txt_indx = 6
        self.corpus = None
        self.pos_features = None
        self.ngram_features = None
        self.other_features = None
        self.ngram_vectorizer = None
        self.pos_vectorizer = None
        self.lexicon = None

    def get_labels(self):
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
                label = int(row[self.label_indx])
                labels.append(label)
        return np.array(labels)

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

    def init_dataset(self):
        labels = self.generate_corpus_get_labels()
        additional_features = self.init_other_features()
        pos_data = self.init_pos_tags_ds()
        ngram_data = self.init_ngrams_datset()
        lexicon = self.init_lexicon_data()
        print(labels.shape)
        print(additional_features.shape)
        print(pos_data.shape)
        print(ngram_data.shape)
        print(lexicon.shape, np.sum(lexicon))
        self.dataset = np.concatenate([ngram_data, pos_data, additional_features, lexicon], axis=1)
        return (self.dataset, labels)

    @not_none('corpus')
    def init_pos_tags_ds(self):
        dataset = []
        pos_vectorizer = TfidfVectorizer(
                tokenizer=None,
                lowercase=False,
                preprocessor=None,
                ngram_range=(1, 3),
                stop_words=None,
                use_idf=False,
                smooth_idf=False,
                norm=None,
                decode_error='replace',
                max_features=5000,
                min_df=5,
                max_df=0.75
        )

        for tweet in self.corpus:
            tweet = self.replace_mentions_urls(tweet)
            tokens = nltk.word_tokenize(tweet)
            tokens_tagged = nltk.pos_tag(tokens)
            # Taking only the part of speech (Word, PartOfSpeech)
            pos_tags = [pos[1] for pos in tokens_tagged if re.match('\w+', pos[1])]
            curr_row = ' '.join(pos_tags)
            dataset.append(curr_row)

        dataset = pos_vectorizer.fit_transform(dataset).toarray()
        vocab = {v:i for i, v in enumerate(pos_vectorizer.get_feature_names())}
        self.pos_features = vocab
        self.pos_vectorizer = pos_vectorizer
        return dataset

    
    def get_other_features(self, tweet):
        tweet = self.replace_mentions_urls(tweet, "URL", "MENTION", "HASHTAG")
            
        url_count = tweet.count("URL")
        mention_count = tweet.count('MENTION')
        hashtag_count = tweet.count("HASHTAG")

        number_of_words = len(re.split(r'[a-z]+', tweet))
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
        polarity = self.vader_analyzer.polarity_scores(tweet)

        return [url_count, mention_count, hashtag_count, number_of_words,\
                number_of_terms, number_of_unique_terms,number_of_syllables,\
                number_of_chars, number_of_chars_total, avrg_syllables,\
                fkra_score, fre_score,\
                polarity['neg'], polarity['neu'], polarity['pos'], polarity['compound']]

    @not_none('corpus')
    def init_other_features(self):
        dataset = []
        for tweet in self.corpus:
            dataset.append(self.get_other_features(tweet))

        self.other_features = ['URLCOUNT', 'MENTIONCOUNT', 'HASHTAGCOUNT', 'WORDSCOUNT',
                               'TERMSCOUNT', 'UNIQUETERMSCOUNT', 'SYLLABLESCOUNT', 'CHARSCOUNT',
                               'TOTALCHARSCOUNT', 'AVRGSYLLABLESCOUNT', 'FKRA', 'FRE',
                               'POLARITYNEG', 'POLARITYNEU', 'POLARITYPOS', 'POLARITYCOMP']

        return np.array(dataset)

    @not_none('corpus')
    def init_ngrams_datset(self, pattern=r"\W+"):
        dataset = []

        ngram_vectorizer = TfidfVectorizer(use_idf=True,
                                           tokenizer=PreprocessHateData.tokenize,
                                           stop_words=nltk.corpus.stopwords.words('english'),
                                           decode_error='replace',
                                           min_df=5,
                                           max_df=0.75,
                                           max_features=10000,
                                           smooth_idf=False,
                                           norm=None,
                                           ngram_range=(1, 3),
                                           token_pattern=r'\b\w+\b')

        for tweet in self.corpus:
            tweet = self.replace_mentions_urls(tweet)
            # Adding the reduced sentence to the dataset(corpus)
            dataset.append(tweet)
        
        dataset = ngram_vectorizer.fit_transform(dataset).toarray()

        vocab = self.ngram_scores = {v:i for i, v in enumerate(ngram_vectorizer.get_feature_names())}
        self.ngram_features = {i:ngram_vectorizer.idf_[i] for i in vocab.values()}
        self.ngram_vectorizer = ngram_vectorizer

        return dataset
 
    @not_none('corpus')
    def init_lexicon_data(self):
        lx = Lexicon([''], ['refined_ngram_dict.csv'],
                     'merinjei_classification/data')
        lx.init_lexicon()
        self.lexicon = lx.get_lexicon()
        dataset = []

        for tweet in self.corpus:
            row = [0 for _ in range(len(self.lexicon))]
            tweet_ngrams = tweet.split() + list(trigrams(tweet.split())) + list(bigrams(tweet.split()))
            for ngram in tweet_ngrams:
                ngram = ' '.join(ngram)
                if ngram in self.lexicon:
                    # Add it to the dataset on its correct position
                    indx = list(self.lexicon.keys()).index(ngram)
                    row[indx] = self.lexicon[ngram]
            dataset.append(row)
        return np.array(dataset)

    def init_features(self):
        self.features = {
            'ngram_scores': self.ngram_scores,
            'ngram_features': self.ngram_features,
            'pos_features': self.pos_features,
            'other_features': self.other_features,
            'lexicon': self.lexicon,
            'ngram_vectorizer': self.ngram_vectorizer,
            'pos_vectorizer': self.pos_vectorizer
        }
        return self.features        

    def replace_mentions_urls(self, tweet, replace_url='', replace_mention='', replace_hashtag=''):
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
        tweet = ' '.join(re.split(r'\s+', tweet))

        return tweet
    
    @staticmethod
    def tokenize(tweet: str) -> list:
        stemmer = SnowballStemmer("english")
        tokens = tweet.split()
        tokens = [stemmer.stem(w) for w in tokens]
        return tokens
