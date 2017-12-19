import time
import pickle
import numpy as np
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score, recall_score, f1_score


def add_to_array(array: list, word: str, val: str, features, stemmer):
    if word in stopwords.words():
        return

    word = stemmer.stem(word)
    if word in features:
        array[features.index(word)] += int(val)

def get_unused_dataset_indxs(dataset, bottom_threshold, top_threshold):
    summed_dataset = np.sum(dataset, axis=0)
    bottom_indxs_delete = summed_dataset <= bottom_threshold
    top_indxs_delete = summed_dataset >= top_threshold
    cols_to_delete = np.logical_or(bottom_indxs_delete, top_indxs_delete)
    indx_to_delete = [indx for indx in range(len(cols_to_delete)) if cols_to_delete[indx]]

    return indx_to_delete


def get_unused_features(features_importance, threshold=0):
    cols_to_delete = features_importance <= threshold
    indx_to_delete = [indx for indx in range(len(cols_to_delete)) if cols_to_delete[indx]]
    return indx_to_delete


def concat_features(features_a, features_b, file):
    features_a = set(features_a)
    features_a.update(features_b)
    features_a = list(features_a)
    with open(file, 'wb') as f:
        pickle.dump(features_a, f)


def split_to_train_test(features_and_labels: list, test_set_percent=0.4, shuffle=True, labels=None) -> tuple:
    if shuffle:
        np.random.shuffle(features_and_labels)
    features = features_and_labels[:, :-1]
    if labels is None:
        labels = np.logical_xor(1, features_and_labels[:, -1:].ravel())
    else:
        labels = labels.ravel()
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


def train_and_log(clf_class, features_train, labels_train, features_test, labels_test,):
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
    from sklearn.naive_bayes import GaussianNB
    train_and_log(GaussianNB, features_train, labels_train, features_test, labels_test)

    ### SVM
    # from sklearn.svm import SVC
    # train_and_log(SVC, features_train, labels_train, features_test, labels_test, features)

    ### DecisionTree
    from sklearn.tree import DecisionTreeClassifier
    train_and_log(DecisionTreeClassifier, features_train, labels_train, features_test, labels_test)

    ### RandomForest
    from sklearn.ensemble import RandomForestClassifier
    train_and_log(RandomForestClassifier, features_train, labels_train, features_test, labels_test)

    from sklearn.ensemble import AdaBoostClassifier
    train_and_log(AdaBoostClassifier, features_train, labels_train, features_test, labels_test)

    from sklearn.ensemble import BaggingClassifier
    train_and_log(BaggingClassifier, features_train, labels_train, features_test, labels_test)

    from sklearn.ensemble import GradientBoostingClassifier
    train_and_log(GradientBoostingClassifier, features_train, labels_train, features_test, labels_test)

    # from sklearn.ensemble import VotingClassifier
    # train_and_log(VotingClassifier, features_train, labels_train, features_test, labels_test, features)
