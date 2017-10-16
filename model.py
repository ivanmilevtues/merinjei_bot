import preprocessing_data
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
import time

def split_to_train_test(feature_and_labels):
    features = feature_and_labels[:, :-1]
    labels = feature_and_labels[:, -1:].ravel()
    return (features[:int(len(features) * 0.4)], features[:-int(len(features) * 0.4)],
            labels[:int(len(labels) * 0.4)], labels[:-int(len(labels) * 0.4)])



def main():
    with open("features.txt", "rb") as fp:
        bag_of_words = pickle.load(fp)

    features_and_labels = preprocessing_data.generate_data_set(bag_of_words)
    np.random.shuffle(features_and_labels)
    features_test, features_train, labels_test, labels_train = split_to_train_test(features_and_labels)

    clf = GaussianNB()
    clf.fit(features_train, labels_train.T)

    pred_test = clf.predict(features_test)
    pred_train = clf.predict(features_train)
    with open("results.txt", "a") as fp:
        fp.write("""=======================
On test data: {}
On train data: {}
Time: {}
=======================""".format(accuracy_score(pred_test, labels_test), accuracy_score(pred_train, labels_train), time.asctime()))


if __name__ == "__main__":
    main()
