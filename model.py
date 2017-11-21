from PreprocessData import PreprocessData
from LineParser import LineParser
from PreprocessHateData import PreprocessHateData
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
import time
from sklearn.metrics import average_precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from preprocessing_utilities import get_unused_dataset_indxs, get_unused_features


def split_to_train_test(features_and_labels: list, test_set_percent=0.4, shuffle=True) -> tuple:
    if shuffle:
        np.random.shuffle(features_and_labels)
    features = features_and_labels[:, :-1]
    labels = features_and_labels[:, -1:].ravel()
    return (features[:int(len(features) * test_set_percent)], features[-int(len(features) * (1. - test_set_percent)):],
            labels[:int(len(labels) * test_set_percent)], labels[-int(len(labels) * (1. - test_set_percent)):])


def log_classifier(clf, train_labels_pred, train_labels_true, test_labels_pred, test_labels_true, time_start, time_end):
    test_acc = accuracy_score(test_labels_true, test_labels_pred)
    train_acc = accuracy_score(train_labels_true, train_labels_pred)

    test_precision = average_precision_score(test_labels_true, test_labels_pred)
    train_precision = average_precision_score(train_labels_true, train_labels_pred)

    test_recall = recall_score(test_labels_true, test_labels_pred, average='binary')
    train_recall = recall_score(train_labels_true, train_labels_pred, average='binary')

    test_f1 = f1_score(test_labels_true, test_labels_pred, average='binary')
    train_f1 = f1_score(train_labels_true, train_labels_pred, average='binary')
    log_text = """=======================
Date: {}
Time taken to train: {} seconds
Classifier: {}
---------------------------------
Test Accuracy   : {}
Train Accuracy  : {}
---------------------------------
Test precision  : {}
Train precision : {}
---------------------------------
Test recall     : {}
Train recall    : {}
---------------------------------
Test f1         : {}
Train f1        : {}
=======================""".format(time.asctime(), time_end - time_start, str(clf),
                                  test_acc, train_acc,
                                  test_precision, train_precision,
                                  test_recall, train_recall,
                                  test_f1, train_f1)
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

    log_classifier(clf, pred_train, labels_train, pred_test, labels_test,
                   time_start, time_end)


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
        print(len(ds), sum(ds[0]))
        print(clf.predict_proba(ds))
        print(clf.predict(ds))


def preprocess_data():
    preprocess = PreprocessHateData("", "")

    preprocess.load_features()
    features = preprocess.get_features()

    preprocess.load_dataset()
    labeled_data = preprocess.get_dataset()

    preprocess.load_dataset("unlabled_dataset.pickle")
    unlabeled_data = preprocess.get_dataset()

    # preprocess.load_dataset("hatespeech_dataset.pickle")
    # hatespeech_data = preprocess.get_dataset()
    # hatespeech_data = preprocess.balance_dataset()

    full_dataset = np.concatenate((labeled_data, unlabeled_data), axis=0)

    features_test, features_train, labels_test, labels_train = split_to_train_test(full_dataset, test_set_percent=0.2,
                                                                                   shuffle=True)

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier()
    clf.fit(features_train, labels_train)

    indx_to_delete = get_unused_features(clf.feature_importances_)
    dataset = PreprocessData.reduce_dataset(full_dataset, indx_to_delete)

    preprocess.dataset = full_dataset
    preprocess.save_dataset('reduced_data_nonhatespeech.pickle')

    preprocess.load_dataset("hatespeech_dataset.pickle")
    hatespeech_data = preprocess.get_dataset()
    hatespeech_data = preprocess.balance_dataset()

    hatespeech_data = PreprocessHateData.reduce_dataset(hatespeech_data, indx_to_delete)

    return np.array(hatespeech_data)


def main():
    preprocess = PreprocessData([],[])
    preprocess.load_dataset("dataset_review_w_reduced_full_features.pickle")
    preprocess.balance_dataset()
    hs_dataset = preprocess.get_dataset()
    preprocess.load_dataset("dataset_hs_w_reduced_full_features.pickle")
    review_datset = preprocess.get_dataset()

    full_dataset = np.concatenate((hs_dataset, review_datset), axis=0)

    features_test, features_train, labels_test, labels_train = split_to_train_test(full_dataset)

    # train_classifiers(features_test, features_train, labels_test, labels_train)

    from sklearn.ensemble import RandomForestClassifier
    time_start = time.time()
    clf = RandomForestClassifier(n_estimators=100 , n_jobs=-1)
    clf.fit(features_train, labels_train)
    time_end = time.time()
    pred_train = clf.predict(features_train)
    pred_test = clf.predict(features_test)
    log_classifier(clf, pred_train, labels_train, pred_test, labels_test,
                   time_start, time_end)
    print(len(full_dataset[0]))
    preprocess.load_features("reduced_full_features.pickle")
    print(len(preprocess.get_features()))
    with open('classifier.pickle', 'wb') as f:
        pickle.dump(clf, f)
    terminal_testing(clf, preprocess.get_features())


if __name__ == "__main__":
    main()