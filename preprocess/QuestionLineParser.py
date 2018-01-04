import numpy as np
from nltk.stem import SnowballStemmer
from preprocess.preprocessing_utilities import add_to_array
from collections import Counter


class QuestionLineParser:

    def __init__(self, features):
        self.features = features
        self.stemmer = SnowballStemmer('english')    

    def parse_line(self, question):
        data = [0 for _ in range(len(self.features))]

        tokens = question.split()
        tokens = Counter(word for word in tokens)
        
        for k, v in tokens.items():
            add_to_array(data, k, v, self.features, self.stemmer)
        
        return np.array([data])
