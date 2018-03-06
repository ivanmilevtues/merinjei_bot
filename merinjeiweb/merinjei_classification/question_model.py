from merinjei_classification.preprocess.PreprocessJSONQuestions import PreprocessJSONQuestions
from merinjei_classification.preprocess.PreprocessQuestions import PreprocessQuestions
from merinjei_classification.preprocess.preprocessing_utilities import split_to_train_test
from merinjei_classification.preprocess.QuestionLineParser import QuestionLineParser
from merinjei_classification.training.train_and_log import train_classifiers, log_classifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
import time
import numpy as np
import pickle


def init_dataset():
    
    pjson_q = PreprocessJSONQuestions(['questions'], ['question_types.json'])
    # pjson_q.init_features()
    # pjson_q.save_features('data/processed_data/questions_json_features.pkl')

    # fs_json_q = pjson_q.load_and_get_features('data/processed_data/questions_json_features.pkl')

    pq = PreprocessQuestions(['questions'], ['question_types01.txt'])   
    # pq.init_features()
    # pq.save_features('data/processed_data/questions_types01_features.pkl')
    # fs_q = pq.load_and_get_features('data/processed_data/questions_types01_features.pkl')
    
    # with open('data/processed_data/questions_full_features.pkl', 'wb') as f:
    #     fs = set(fs_json_q)
    #     fs.update(fs_q)
    #     pickle.dump(list(fs), f)
    
    # pq.load_features('data/processed_data/questions_full_features.pkl')
    # pjson_q.load_features('data/processed_data/questions_full_features.pkl')
    # print(pjson_q.features)

    # ds_json_q = pjson_q.init_dataset()
    ds_json_q = pjson_q.load_and_get_dataset(
        'merinjei_classification/data/processed_data/dataset_questions_json_full_fs.pkl')


    # ds_q = pq.init_dataset()
    ds_q = pq.load_and_get_dataset(
        'merinjei_classification/data/processed_data/dataset_questions_types_full_fs.pkl')

    return np.concatenate([ds_q, ds_json_q])


def quesition_model_init():
    ds = init_dataset()
    pq = PreprocessQuestions([''], [''])
    features_test, features_train, labels_test, labels_train = split_to_train_test(ds, test_set_percent=0.2)
    
    clf = RandomForestClassifier()
    clf.fit(features_train, labels_train)
    return clf


def main():
    ds = init_dataset()
    
    pq = PreprocessQuestions([''], [''])
    fs = pq.load_and_get_features('data/processed_data/questions_full_features.pkl')
    features_test, features_train, labels_test, labels_train = split_to_train_test(ds, test_set_percent=0.2)
    # train_classifiers(features_test, features_train, labels_test, labels_train)
    
    time_start = time.time()
    clf = RandomForestClassifier()

    clf.fit(features_train, labels_train)
    time_end = time.time()
    
    pred_train = clf.predict(features_train)
    pred_test = clf.predict(features_test)
    
    log_classifier(clf, pred_train, labels_train, pred_test, labels_test,
                   time_start, time_end)
    

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
