# The purpose of this file is to open
# and parse the data into numpy

from pprint import pprint
from collections import Counter
import numpy as np
import re
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


main_directory = "processed_acl"
sub_directories = ["books"]
data_types = ["negative", "positive"]


def generate_bag_of_words() -> list:
    parsed_data = set()
    for sub_dir in sub_directories:
        for data_type in data_types:
            with(open(main_directory + "/" + sub_dir + "/" + data_type + ".review")) as f:
                parsed_data.update(parse_to_unique_words(f.readlines()))
    return list(parsed_data)


def parse_to_unique_words(data_string: str) -> set():
    result = set()
    for row in data_string:
        result.update(set(re.split(':\d|\s', row)))
    return result


def generate_data_set(word_bag: list) -> tuple:
    features = []
    labels = []
    for sub_dir in sub_directories:
        for data_type in data_types:
            with(open(main_directory + "/" + sub_dir + "/" + data_type + ".review")) as f:
                curr_features, curr_labels = data_to_numpy_array(f.readlines(), word_bag, 1 if data_type == 'positive' else 0)
                features += curr_features
                labels += curr_labels
    return (np.array(features), np.array(labels))


def data_to_numpy_array(data_string: str, bag_of_words: list, label_data: int) -> list:
    result = []
    labels = []
    for index in range(int(len(data_string) * 0.2)):
        row = data_string[index]
        result_row = [0 for i in range(len(bag_of_words))]
        items = re.split(":|\s", row)
        for i in range(0, len(items) - 2, 2):
            try:
                result_row[bag_of_words.index(items[i])] += int(items[i+1])
            except Exception:
                continue
        result.append(result_row)
        labels.append(label_data)
    return (result, labels)

if __name__ == "__main__":
    bag_of_words = generate_bag_of_words()
    features , labels = generate_data_set(bag_of_words)
    features_train = features[:300]
    labels_train = labels[:300]
    features_test = features[300:]
    labels_test = labels[300:]
    print(features_train.shape, labels_train.T.shape)
    clf = GaussianNB()
    clf.fit(features_train, labels_train.T)
    pred = clf.predict(features_test)
    pred1 = clf.predict(features_train)
    print(accuracy_score(labels_train, pred1))
    for i in pred:
        print(pred)
    for i in labels_test:
        print(labels_test)
    print(accuracy_score(labels_test, pred))

