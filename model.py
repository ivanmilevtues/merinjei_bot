from data_preprocessing import PreprocessData
from live_line_parser import LineParser
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
import time
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from preprocessing_utilities import get_unused_dataset_indxs, get_unused_features


def split_to_train_test(features_and_labels: list, test_set_percent=0.4, shuffle=True) -> tuple:
    if shuffle:
        np.random.shuffle(features_and_labels)
    features = features_and_labels[:, :-1]
    labels = features_and_labels[:, -1:].ravel()
    return (features[:int(len(features) * test_set_percent)], features[-int(len(features) * (1.-test_set_percent)):],
            labels[:int(len(labels) * test_set_percent)], labels[-int(len(labels) * (1. - test_set_percent)):])


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
    print(clf_class.__name__ + " has started training.")
    clf = clf_class()
    time_start = time.time()
    clf.fit(features_train, labels_train.T)
    time_end = time.time()
    pred_test = clf.predict(features_test)
    pred_train = clf.predict(features_train)
    precision = average_precision_score(pred_test, labels_test)
    log_classifier(clf, accuracy_score(pred_test, labels_test), accuracy_score(pred_train, labels_train),
                   time_start, time_end, precision)


def train_classifiers(features_test, features_train, labels_test, labels_train):
    ## Logistic regresion
    from sklearn.linear_model import LogisticRegression
    train_and_log(LogisticRegression, features_train, labels_train, features_test, labels_test)

    ### Nayve bayes
    train_and_log(GaussianNB, features_train, labels_train, features_test, labels_test)

    ### SVM
    # from sklearn.svm import SVC
    # train_and_log(SVC, features_train, labels_train, features_test, labels_test)

    ### DecisionTree
    from sklearn import tree
    train_and_log(tree.DecisionTreeClassifier, features_train, labels_train, features_test, labels_test)

    ### RandomForest
    from sklearn.ensemble import RandomForestClassifier
    train_and_log(RandomForestClassifier, features_train, labels_train, features_test, labels_test)


def plot(data):
    x = list(range(len(data)))
    y = data

    plt.plot(x, y, 'ro')
    plt.axis([0, len(x), min(y), max(y)])

    plt.xlabel('feature number')
    plt.ylabel('feature importance')
    plt.title('Features importance rate')

    plt.show()


def terminal_testing(clf, features):
    lp = LineParser(features)

    for _ in range(10):
        inp = input("Your hate here:")
        ds = lp.parse_line(inp)
        print(clf.predict_proba(ds))
        print(clf.predict(ds))
        print('-----------')


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

    features_test, features_train, labels_test, labels_train = split_to_train_test(full_dataset, test_set_percent=0.2, shuffle=True)

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier()
    clf.fit(features_train, labels_train)

    indx_to_delete = get_unused_features(clf.feature_importances_)
    dataset = PreprocessData.reduce_dataset(full_dataset, indx_to_delete)
    features_test, features_train, labels_test, labels_train = split_to_train_test(dataset, test_set_percent=0.2, shuffle=True)

    print(features_test.shape, features_train.shape, labels_train.shape, labels_test.shape)
    # train_classifiers(features_test, features_train, labels_test, labels_train)

    clf = RandomForestClassifier()
    clf.fit(features_train, labels_train)

    features = PreprocessData.reduce_features(features, indx_to_delete)
    terminal_testing(clf, features)
    # plot(clf.feature_importances_ )



if __name__ == "__main__":
    main()
