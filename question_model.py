from preprocess.PreprocessQuestions import PreprocessQuestions
from preprocess.PreprocessData import PreprocessData
from preprocess.PreprocessCocoQuestions import PreprocessCocoQuestions
from preprocess.preprocessing_utilities import split_to_train_test, train_classifiers


def init_dataset():
    # pq = PreprocessCocoQuestions(['questions'], ['questions.txt'])
    # pq.init_features()
    # pq.load_features('data/processed_data/questions_cocoa_featues.pickle')
    # ds = pq.init_dataset()
    # pq.save_dataset('data/processed_data/dataset_questions_cocoa.pickle')
    pq = PreprocessQuestions(['questions'], ['question_types01.txt'])
    pq.load_features('data/processed_data/questions_full_features.pickle')
    ds = pq.init_dataset()
    pq.save_dataset('dataset_questions_header_labels.pickle')
    pq.save_labels()
    return ds

def init_features():
    pq = PreprocessQuestions(['questions'], ['question_types01.txt'])
    pq.init_features()
    pq.save_features('data/processed_data/questions_full_features.pickle')

def main():
    ds = init_dataset()
    features_test, features_train, labels_test, labels_train = split_to_train_test(ds, test_set_percent=0.2,
                                                                                   shuffle=True)
    train_classifiers(features_test, features_train, labels_test, labels_train)
    


if __name__ == '__main__':
    main()
