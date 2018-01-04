import time
from sklearn.metrics import classification_report

def log_classifier(clf, train_labels_pred, train_labels_true, test_labels_pred, test_labels_true, time_start, time_end):
    log_text = """=======================
Date: {}
Time taken to train: {} seconds
Classifier: {}
---------------------------------
Test:
{}
Train:
{}
=======================""".format(time.asctime(), time_end - time_start, str(clf),
                                  classification_report(test_labels_pred, test_labels_true),
                                  classification_report(train_labels_pred, train_labels_true))
    with open("resultsQuestions.txt", "a") as f:
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
    ## Logistic regresions
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

    # from sklearn.ensemble import GradientBoostingClassifier
    # train_and_log(GradientBoostingClassifier, features_train, labels_train, features_test, labels_test)

    # from sklearn.ensemble import VotingClassifier
    # train_and_log(VotingClassifier, features_train, labels_train, features_test, labels_test, features)

