import numpy as np
import re
import nltk
from preprocess.preprocessing_utilities import add_to_array
from collections import Counter
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfTransformer
from preprocess.PreprocessHateData import PreprocessHateData


class LineParser():

    def __init__(self, features):
        self.features = features
        self.pos_analyzer = self.features['ngram_vectorizer'].build_analyzer()
        self.ngram_analyzer = self.features['pos_vectorizer'].build_analyzer()
        self.pphd = PreprocessHateData([], [])

    def parse_line(self, tweet):
        dataset = [0 for _ in range(len(self.features['ngram_features']) + 
                               len(self.features['pos_features']) + 
                               len(self.features['other_features']))]


        cleaned_tweet = self.pphd.replace_mentions_urls(tweet)

        pos = nltk.pos_tag(cleaned_tweet)
        pos = ' '.join(pos_tag for _, pos_tag in pos if re.match('\w+', pos_tag))
        pos_ngrams = self.ngram_analyzer(pos)

        ngrams = self.ngram_analyzer(cleaned_tweet)

        for pos_ngram in pos_ngrams:
            try:
                indx = self.features['ngram_features'][pos_ngram]
                dataset[indx] += 1
            except KeyError:
                continue
        
        for ngram in ngrams:
            try:
                indx = self.features['ngrams_features'][ngram]
                score = self.features['ngram_score'][indx]
                dataset[indx] += score
            except KeyError:
                continue
        
        other_features = self.pphd.get_other_features(tweet)
        for indx in range(len(other_features)):
            ds_indx = (len(other_features) - indx)
            dataset[-ds_indx] = other_features[indx]
        
        return np.array([dataset])
