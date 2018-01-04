from preprocess.PreprocessJSONQuestions import PreprocessJSONQuestions
from preprocess.PreprocessQuestions import PreprocessQuestions
from preprocess.preprocessing_utilities import split_to_train_test
from preprocess.QuestionLineParser import QuestionLineParser
from training.train_and_log import train_classifiers
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
import time
import numpy as np


def init_dataset():
    
    pq = PreprocessJSONQuestions(['questions'], ['question_types.json'])
    # pq.load_features('data/processed_data/questions_full_features.pkl')
    # ds_json = pq.init_dataset()
    ds_json = pq.load_and_get_dataset('data/processed_data/dataset_qeustions_json_full_fs.pkl')
    pq = PreprocessQuestions(['questions'], ['question_types01.txt'])
    # pq.load_features('data/processed_data/questions_full_features.pkl')
    # ds_que = pq.init_dataset()
    ds_que = pq.load_and_get_dataset('data/processed_data/dataset_qeustions_types_full_fs.pkl')
    return np.concatenate([ds_json, ds_que])


def main():
    ds = init_dataset()
    
    pq = PreprocessQuestions([''], [''])
    fs = pq.load_and_get_features('data/processed_data/questions_full_features.pkl')
    features_test, features_train, labels_test, labels_train = split_to_train_test(ds, test_set_percent=0.2)
    # train_classifiers(features_test, features_train, labels_test, labels_train)
    
    time_start = time.time()
    clf = LogisticRegression()

    clf.fit(features_train, labels_train)
    time_end = time.time()
    
    pred_train = clf.predict(features_train)
    pred_test = clf.predict(features_test)
    
    # log_classifier(clf, pred_train, labels_train, pred_test, labels_test,
    #                time_start, time_end)
    
    print(classification_report(pred_test, labels_test))
    print(features_test.shape)
    del features_test
    del features_train
    del labels_test
    del labels_train

    lp = QuestionLineParser(fs)
    for _ in range(10):
        sentence = input('Question> ')
        result = lp.parse_line(sentence)
        print(result.shape)
        print(clf.predict(result))
        print(['ABBR', 'DESC', 'PROCEDURE', 'PERSON', 'LOCATION', 'NUMBER', 'ORGANIZATION', 'CAUSALITY'][clf.predict(result)[0]])
        print(clf.predict_proba(result))

if __name__ == '__main__':
    
    main()
