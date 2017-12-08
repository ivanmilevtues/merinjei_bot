import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import  TfidfTransformer
from sklearn.metrics import average_precision_score, recall_score, f1_score
from data.failed_test_examples import FAILED_EXAMPLES
from preprocess.LineParser import LineParser
from preprocess.PreprocessData import PreprocessData
from preprocess.PreprocessHateData import PreprocessHateData
from preprocess.preprocessing_utilities import get_unused_dataset_indxs, get_unused_features


def split_to_train_test(features_and_labels: list, test_set_percent=0.4, shuffle=True) -> tuple:
    if shuffle:
        np.random.shuffle(features_and_labels)
    features = features_and_labels[:, :-1]
    labels = np.logical_xor(1, features_and_labels[:, -1:].ravel())
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


def train_and_log(clf_class, features_train, labels_train, features_test, labels_test, features):
    print(clf_class.__name__ + " has started training.")
    clf = clf_class()
    time_start = time.time()
    clf.fit(features_train, labels_train.T)
    time_end = time.time()
    pred_test = clf.predict(features_test)
    pred_train = clf.predict(features_train)
    lp = LineParser(features)
    for example in FAILED_EXAMPLES:
        ds = lp.parse_line(example)
        print(str(clf.predict_proba(ds)) + ' <- ' + example)

    log_classifier(clf, pred_train, labels_train, pred_test, labels_test,
                   time_start, time_end)


def train_classifiers(features_test, features_train, labels_test, labels_train, features):
    ## Logistic regresion
    from sklearn.linear_model import LogisticRegression
    train_and_log(LogisticRegression, features_train, labels_train, features_test, labels_test, features)

    ### Nayve bayes
    train_and_log(GaussianNB, features_train, labels_train, features_test, labels_test, features)

    ### SVM
    # from sklearn.svm import SVC
    # train_and_log(SVC, features_train, labels_train, features_test, labels_test, features)

    ### DecisionTree
    from sklearn import tree
    train_and_log(tree.DecisionTreeClassifier, features_train, labels_train, features_test, labels_test, features)

    ### RandomForest
    from sklearn.ensemble import RandomForestClassifier
    train_and_log(RandomForestClassifier, features_train, labels_train, features_test, labels_test, features)

    from sklearn.ensemble import AdaBoostClassifier
    train_and_log(AdaBoostClassifier, features_train, labels_train, features_test, labels_test, features)

    from sklearn.ensemble import BaggingClassifier
    train_and_log(BaggingClassifier, features_train, labels_train, features_test, labels_test, features)

    from sklearn.ensemble import GradientBoostingClassifier
    train_and_log(GradientBoostingClassifier, features_train, labels_train, features_test, labels_test, features)

    # from sklearn.ensemble import VotingClassifier
    # train_and_log(VotingClassifier, features_train, labels_train, features_test, labels_test, features)


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
        print(sum(ds[0]))
        print(clf.predict_proba(ds))
        print(clf.predict(ds))


def preprocess_data():
    preprocess = PreprocessData("", "")
    
    preprocess.load_features("./data/processed_data/reduced_full_features.pickle")
    features = preprocess.get_features()

    preprocess.load_dataset("./data/processed_data/dataset_review_w_reduced_full_features.pickle")
    preprocess.balance_dataset()
    hs_dataset = preprocess.get_dataset()

    preprocess.load_dataset("./data/processed_data/dataset_hs_w_reduced_full_features.pickle")
    review_datset = preprocess.get_dataset()

    full_dataset = np.concatenate((hs_dataset, review_datset), axis=0)
    print(full_dataset.shape)
    dataset = full_dataset[:, :-1]
    labels = full_dataset[:, -1:]

    unused_indxs = get_unused_dataset_indxs(full_dataset, 10, int(len(full_dataset) * 0.3))
    
    full_dataset = PreprocessData.reduce_dataset(dataset, unused_indxs)
    features = PreprocessData.reduce_features(features, unused_indxs)

    full_dataset = np.append(full_dataset, labels, axis=1)
    print(full_dataset.shape)
    features_test, features_train, labels_test, labels_train = split_to_train_test(full_dataset)
    
    features_test  = TfidfTransformer().fit_transform(features_test).toarray()
    features_train = TfidfTransformer().fit_transform(features_train).toarray()

    features_test, features_train, labels_test, labels_train = split_to_train_test(full_dataset, test_set_percent=0.2,
                                                                                   shuffle=True)

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier()
    clf.fit(features_train, labels_train)

    indx_to_delete = get_unused_features(clf.feature_importances_ )
    full_dataset = PreprocessData.reduce_dataset(full_dataset, indx_to_delete)
    features = PreprocessData.reduce_features(features, indx_to_delete)

    return (full_dataset, features)


def main():
    full_dataset, features = preprocess_data()
    
    features_test, features_train, labels_test, labels_train =\
        split_to_train_test(full_dataset, test_set_percent=0.4)

    print(len(features_test), len(features_train))
    train_classifiers(features_test, features_train, labels_test, labels_train, features)

    from sklearn.ensemble import RandomForestClassifier
    time_start = time.time()
    
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    clf.fit(features_train, labels_train)
    time_end = time.time()
    
    pred_train = clf.predict(features_train)
    pred_test = clf.predict(features_test)
    
    log_classifier(clf, pred_train, labels_train, pred_test, labels_test,
                   time_start, time_end)
    plot(clf.feature_importances_)

    with open('classifier.pickle', 'wb') as f:
        pickle.dump(clf, f)
    
    terminal_testing(clf, features)


if __name__ == "__main__":
    main()