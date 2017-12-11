from preprocess.PreprocessQuestions import PreprocessQuestions
from preprocess.PreprocessData import PreprocessData
from preprocess.preprocessing_utilities import split_to_train_test, train_classifiers

def init_features():
    pq = PreprocessQuestions(['questions'], ['question_types01.txt'])
    pq.init_features()
    pq.save_features('data/processed_data/questions_full_features.pickle')

def main():
    pd = PreprocessData([''], [''])
    ds = pd.load_and_get_dataset('data/processed_data/dataset_questions_full_features.pickle')
    features_test, features_train, labels_test, labels_train = split_to_train_test(ds, test_set_percent=0.2,
                                                                                   shuffle=True)
    train_classifiers(features_test, features_train, labels_test, labels_train)
    


if __name__ == '__main__':
    main()