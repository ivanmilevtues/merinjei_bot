# The purpose of this file is to open
# and parse the data into numpy

import pickle
import re
import numpy as np
from nltk.corpus import stopwords

main_directory = "data"
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
        result.update(re.split(':\d|\s', row))
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


rows_sum = []

def data_to_numpy_array(data_string: str, bag_of_words: list, label_data: int) -> list:
    result = []
    labels = []
    for index in range(int(len(data_string)/10)):
        row = data_string[index]
        print(index, len(data_string)/ 10)
        result_row = [0 for i in range(len(bag_of_words))]
        items = re.split(":|\s", row)
        for i in range(0, len(items) - 2, 2):
            try:
                result_row[bag_of_words.index(items[i])] += int(items[i+1])
                rows_sum[bag_of_words.index(items[i])] += int(items[i+1])
            except Exception:
                continue
        result.append(result_row)
        labels.append(label_data)
    return (result, labels)

def remove_one_words():
    bag_of_words = generate_bag_of_words()
    rows_sum = [0 for i in range(len(bag_of_words))]
    features, labels = generate_data_set(bag_of_words)
    real_features = []
    for ind in range(len(rows_sum)):
        if ind > 50:
            real_features.append(bag_of_words[ind])

    with open("features.txt", "wb") as f:
        pickle.dump(real_features, f)


def remove_stop_words(file="features.txt"):
    with open(file, "rb") as f:
        features = pickle.load(f)
    print(len(features))

    for feature in features:
        if feature in stopwords.words():
            features.remove(feature)
    print(len(features))
    with open(file, "wb") as f:
        pickle.dump(features, f)

if __name__ == "__main__":
    remove_stop_words()