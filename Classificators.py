import pickle
from model import hatespeech_model_init
from question_model import quesition_model_init


class Classificators:
    
    def __init__(self):
        self.hs_classifier = None
        self.question_classifer = None

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


if __name__ == '__main__':
    clfs = Classificators()
    clfs.init_hatespeech_classifier()
    clfs.save_hatespeech_classifer()
    clfs.init_question_classifier()
    clfs.save_question_classifier()