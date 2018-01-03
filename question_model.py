from preprocess.PreprocessQuestions import PreprocessQuestions
from preprocess.PreprocessData import PreprocessData
from preprocess.PreprocessCocoQuestions import PreprocessCocoQuestions
from preprocess.preprocessing_utilities import split_to_train_test
from training.train_and_log import train_classifiers


def init_dataset():
    pq = PreprocessQuestions(['questions'], ['question_types01.txt'])
    pq.init_features()
    pq.load_features('data/processed_data/questions_full_features.pkl')
    # pq.init_dataset()
    # pq.save_dataset('data/processed_data/dataset_questions_full_features.pkl')
    # ds = pq.get_dataset()
    return pq.load_and_get_dataset('data/processed_data/dataset_questions_full_features.pkl')


def main():
    ds = init_dataset()
    print(ds.shape)
    # pq = PreprocessQuestions(['questions'], ['question_types01.txt'])
    # ds = pq.load_and_get_dataset('data/processed_data/dataset_questions_full_features.pkl')
    features_test, features_train, labels_test, labels_train = split_to_train_test(ds, test_set_percent=0.2,
                                                                                   shuffle=True)
    print(labels_train.shape)
    print(labels_test.shape)
    print(type(labels_train))
    print(labels_train)
    train_classifiers(features_test, features_train, labels_test, labels_train)
    


if __name__ == '__main__':
    
    main()
