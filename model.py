from data_preprocessing import PreprocessData
from live_line_parser import LineParser
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
import time
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt


def split_to_train_test(features_and_labels: list, test_set_percent=0.4, shuffle=True) -> tuple:
    if shuffle:
        np.random.shuffle(features_and_labels)
    features = features_and_labels[:, :-1]
    labels = features_and_labels[:, -1:].ravel()
    return (features[:int(len(features) * test_set_percent)], features[:-int(len(features) * test_set_percent)],
            labels[:int(len(labels) * test_set_percent)], labels[:-int(len(labels) * test_set_percent)])


def log_classifier(clf, test_set_acc, train_set_acc, time_start, time_end, precision):
    log_text = """=======================
On test data: {}
On train data: {}
Date: {}
Time taken to train: {} seconds
Classifier: {}
Precision: {}
=======================""".format(test_set_acc, train_set_acc, time.asctime(), time_start - time_end,
                                  str(clf), precision)
    with open("classifier.txt", "a") as fp:
        fp.write(log_text)
    with open("results.txt", "a") as f:
        f.write(log_text)


def train_and_log(clf_class, features_train, labels_train, features_test, labels_test):
    clf = clf_class()
    time_start = time.time()
    clf.fit(features_train, labels_train.T)
    time_end = time.time()
    pred_test = clf.predict(features_test)
    pred_train = clf.predict(features_train)
    precision = average_precision_score(pred_test, labels_test)
    log_classifier(clf, accuracy_score(pred_test, labels_test), accuracy_score(pred_train, labels_train),
                   time_start, time_end, precision)


def train_classifiers(features_train, labels_train, features_test, labels_test):
    ## Logistic regresion
    from sklearn.linear_model import LogisticRegression
    train_and_log(LogisticRegression, features_train, labels_train, features_test, labels_test)

    ### Nayve bayes
    train_and_log(GaussianNB, features_train, labels_train, features_test, labels_test)

    ### SVM
    from sklearn.svm import SVC
    train_and_log(SVC, features_train, labels_train, features_test, labels_test)

    ### DecisionTree
    from sklearn import tree
    train_and_log(tree.DecisionTreeClassifier, features_train, labels_train, features_test, labels_test)

    ### RandomForest
    from sklearn.ensemble import RandomForestClassifier
    train_and_log(RandomForestClassifier, features_train, labels_train, features_test, labels_test)


def main():
    with open("features.txt", "rb") as fp:
        bag_of_words = pickle.load(fp)
    preprocess = PreprocessData("", "")

    preprocess.load_features()
    features = preprocess.get_features()

    preprocess.load_dataset()
    labeled_data = preprocess.get_dataset()

    preprocess.load_dataset("unlabled_dataset.pickle")
    unlabeled_data = preprocess.get_dataset()

    full_dataset = np.concatenate((labeled_data, unlabeled_data), axis=0)

    # dataset = PreprocessData.reduce_dataset(full_dataset)
    # features = PreprocessData.reduce_features(full_dataset, features)

    features_test, features_train, labels_test, labels_train = split_to_train_test(full_dataset, test_set_percent=0.4, shuffle=True)

    # print(features)

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier()
    clf.fit(features_train, labels_train)

    lp = LineParser(features)

    for _ in range(10):
        inp = input("Your hate here:")
        ds = lp.parse_line(inp)
        print(clf.predict_proba(ds))
        print(clf.predict(ds))
        print('-----------')

if __name__ == "__main__":
    main()
