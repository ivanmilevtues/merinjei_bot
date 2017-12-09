from preprocess.PreprocessQuestions import PreprocessQuestions


def init_features():
    pq = PreprocessQuestions(['questions'], ['question_types01.txt'])
    pq.init_features()
    pq.save_features('data/processed_data/questions_full_features.pickle')

def main():
    pq = PreprocessQuestions(['questions'], ['question_types01.txt'])
    pq.load_features('data/processed_data/questions_full_features.pickle')
    pq.init_dataset()
    pq.save_dataset('data/processed_data/dataset_questions_full_features.pickle')
    print(pq.labels)
    print(len(pq.labels))

if __name__ == '__main__':
    main()