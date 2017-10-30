from data_preprocessing import PreprocessData
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
import time
from sklearn.metrics import average_precision_score


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
    clf.fit(features_train, labels_train)
    time_start = time.time()
    clf.fit(features_train, labels_train.T)
    time_end = time.time()
    try:
        k = 0
        for i in clf.coef_:
            if i > 0.5:
                k+=1
                print(k)
    except:
        print(str(clf) + " has no COEF method")
    pred_test = clf.predict(features_test)
    pred_train = clf.predict(features_train)

    precision = average_precision_score(pred_test, labels_test)
    log_classifier(clf, accuracy_score(pred_test, labels_test), accuracy_score(pred_train, labels_train),
                   time_start, time_end, precision)


def main():
    with open("features.txt", "rb") as fp:
        bag_of_words = pickle.load(fp)
    preprocess = PreprocessData("", "")
    preprocess.load_dataset()
    features_and_labels =  preprocess.get_dataset()
    features_test, features_train, labels_test, labels_train = split_to_train_test(features_and_labels, test_set_percent=0.4, shuffle=True)

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

if __name__ == "__main__":
    main()
