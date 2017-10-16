import preprocessing_data
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def main():
    with open("features.txt", "rb") as fp:
        bag_of_words = pickle.load(fp)

    features, labels = preprocessing_data.generate_data_set(bag_of_words)
    features_train = features[:300]
    labels_train = labels[:300]
    features_test = features[300:]
    labels_test = labels[300:]
    print(features_train.shape, labels_train.T.shape)
    clf = GaussianNB()
    clf.fit(features_train, labels_train.T)
    pred = clf.predict(features_test)
    pred1 = clf.predict(features_train)

    with open("results.txt", "w") as fp:
        fp.write("""
On test data: {}
On train data: {}
        """.format(accuracy_score(pred, labels_test), accuracy_score(pred1, labels_train)))


if __name__ == "__main__":
    main()
