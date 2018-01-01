import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import  TfidfTransformer
from data.failed_test_examples import FAILED_EXAMPLES
from preprocess.LineParser import LineParser
from preprocess.PreprocessData import PreprocessData
from preprocess.PreprocessHateData import PreprocessHateData
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from preprocess.preprocessing_utilities import get_unused_dataset_indxs, get_unused_features,\
                                               split_to_train_test, train_classifiers, log_classifier
from preprocess.AutoCorrect import AutoCorrect
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel


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
    
    preprocess.load_features(
        "./data/processed_data/reduced_full_features.pickle")
    features = preprocess.get_features()

    hs_dataset = preprocess.load_and_get_dataset(
        "./data/processed_data/dataset_review_w_reduced_full_features.pickle")

    review_dataset = preprocess.load_and_get_dataset(
        "./data/processed_data/dataset_review_w_reduced_full_features.pickle")

    full_dataset = np.concatenate((hs_dataset, review_dataset), axis=0)
    # full_dataset = hs_dataset

    preprocess.dataset = full_dataset
    preprocess.balance_dataset()
    print(full_dataset.shape)
    print(preprocess.get_dataset().shape)
    full_dataset = preprocess.get_dataset()
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
    preprocess = PreprocessData("", "")

    # we call todense so that we can transform the sparse scipi matrix to numpy matrix
    dataset = preprocess.load_and_get_dataset(
        'dataset_hs_w_trigrams_stemmed.pkl')
    # dataset = dataset.todense()
    # dataset = dataset.A.astype(np.int8)

    labels = preprocess.load_and_get_dataset('labels.pkl')# .astype(np.int8)
    labels = np.array(labels)
    # full_dataset = np.concatenate((dataset, labels), axis=1)
    
    features_test, features_train, labels_test, labels_train =\
        split_to_train_test(dataset, test_set_percent=0.4, shuffle=False, labels=labels)

    # print(features_test.shape, labels_test.shape)
    # print(features_train.shape, labels_train.shape)
    # train_classifiers(features_test, features_train, labels_test, labels_train)

    time_start = time.time()
    pipe = Pipeline(
        [('select', SelectFromModel(LogisticRegression(class_weight='balanced',
                                                       penalty="l1", C=0.01))),
         ('model', LogisticRegression(class_weight='balanced', penalty='l2'))])

    param_grid = [{}]  # Optionally add parameters here

    clf = GridSearchCV(pipe,param_grid, cv=StratifiedKFold(n_splits=5,
                                        random_state=42).split(features_train, labels_train),
                                        verbose=2)

    clf.fit(features_train, labels_train)
    time_end = time.time()
    
    pred_train = clf.predict(features_train)
    pred_test = clf.predict(features_test)
    
    # log_classifier(clf, pred_train, labels_train, pred_test, labels_test,
    #                time_start, time_end)
    
    print(classification_report(pred_test, labels_test))

    del features_test
    del features_train
    del labels_test
    del labels_train

    with open('classifier.pickle', 'wb') as f:
        pickle.dump(clf, f)
    
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    analyzer = vectorizer.build_analyzer()
    for _ in range(10):
    # terminal_testing(clf, features)
        a = input()
        fs = analyzer(a)
        print(clf.predict(fs))
        print(clf.predict_proba(fs))

if __name__ == "__main__":
    # pd = PreprocessHateData(
    #     [''], ['twitter_hate_speech.csv'])
    # # pd.load_features('reduced_full_features.pickle')
    # _, labels = pd.init_dataset()
    # with open('labels.pkl', 'wb') as f:
    #     pickle.dump(labels, f)
    # pd.save_dataset("dataset_hs_w_trigrams_stemmed.pkl")
    main()
