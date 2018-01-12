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
        except FileNotFoundError as e:
            print("One of your files for classifier was not loaded properly! {}".format(e))
        
        try:
            hs_features = self.__load_file(hs_features_path)
            question_features = self.__load_file(question_features_path)
            self.hlp = HateLineParser(hs_features)
            self.qlp = QuestionLineParser(question_features)
        except FileNotFoundError as e:
            print("Could not open feature file! {}".format(e))


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
        data = self.qlp.parse_line(question)
        return self.question_classifer.predict(data)

    def predict_tweet_type(self, tweet):
        data = self.hlp.parse_line(tweet)
        return self.hs_classifier.predict(data)


if __name__ == '__main__':
    clfs = Classificators("hatespeech_clf.pkl", "features.pickle" ,"question_clf.pkl" ,"data/processed_data/questions_full_features.pkl")
    for _ in range(10):
        question = clfs.predict_question_type(input("Ask me:"))
        hate = clfs.predict_tweet_type(input("Hate me:"))
        print("Question")
        print(question)
        print("Hate")
        print(hate)
