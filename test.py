from merinjei_classification.Classifiers import Classifiers

def main():
    clf  = Classifiers("merinjei_classification/classifiers/hatespeech_clf.pkl", "features.pkl",
                       "merinjei_classification/classifiers/question_clf.pkl", "merinjei_classification/data/features/questions_full_features.pkl")
    a = clf.predict_question_type(input("> "))
    b = clf.predict_tweet_type(input("> "))
    print(a, b)

if __name__ == '__main__':
    main()
    