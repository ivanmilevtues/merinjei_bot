import pickle
from model import hatespeech_model_init
from question_model import quesition_model_init
from preprocess.HateLineParser import HateLineParser
from preprocess.QuestionLineParser import QuestionLineParser


class Classificators:
    
    def __init__(self, hs_classifier_path, hs_features_path, question_classifier_path, question_features_path):
        self.hs_classifier = None
        self.question_classifer = None
        self.hlp = None
        self.qlp = None
        hs_features = None
        question_features = None
        try:
            self.question_classifer = self.__load_file(question_classifier_path)
            self.hs_classifier = self.__load_file(hs_classifier_path)
            hs_features = self.__load_file(hs_features_path)
            question_features = self.__load_file(question_features_path)
        except:
            print("Could not load smth")
        self.hlp = HateLineParser(hs_features)
        self.qlp = QuestionLineParser(question_features)

    def __load_file(self, path):
        with open(path, 'rb') as f:
            classifier = pickle.load(f)
        return classifier

    def __save_clf_file(self, path, obj):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    def save_hatespeech_classifer(self, path="hatespeech_clf.pkl"):
        self.__save_clf_file(path, self.hs_classifier)
    
    def save_question_classifier(self, path="question_clf.pkl"):
        self.__save_clf_file(path, self.question_classifer)

    def load_hatespeech_classifier(self, path="hatespeech_clf.pkl"):
        self.hs_classifier = self.__load_file(path)
        return self.hs_classifier

    def load_question_classifier(self, path="question_clf.pkl"):
        self.question_classifer = self.__load_file(path)
        return self.question_classifer

    def init_question_classifier(self):
        self.question_classifer = quesition_model_init()
        return self.question_classifer

    def init_hatespeech_classifier(self):
        self.hs_classifier = hatespeech_model_init()
        return self.hs_classifier

    def predict_question_type(self, question):
        data = self.qlp(question)
        return self.question_classifer.predict(data)

    def predict_tweet_type(self, tweet):
        data = self.hlp(tweet)
        return self.hs_classifier(data)


if __name__ == '__main__':
    clfs = Classificators("", "" ,"" ,"")
    clfs.init_hatespeech_classifier()
    clfs.save_hatespeech_classifer()
    clfs.init_question_classifier()
    clfs.save_question_classifier()