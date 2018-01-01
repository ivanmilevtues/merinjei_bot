import numpy as np
import re
import nltk
from preprocess.preprocessing_utilities import add_to_array
from collections import Counter
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfTransformer


class LineParser():

    def __init__(self, features):
        self.features = features
        self.pos_analyzer = self.features['ngram_vectorizer'].build_analyzer()
        self.ngram_analyzer = self.features['pos_vectorizer'].build_analyzer()

    def parse_line(self, tweet):
        dataset = [0 for _ in range(len(self.features['ngram_features']) + 
                               len(self.features['pos_features']) + 
                               len(self.features['other_features']))]

