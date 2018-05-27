import json
import nltk
import html2text
import numpy as np
import requests
import re
from pprint import pprint
from rake_nltk import Rake
from collections import OrderedDict
from gensim.summarization import summarize
from merinjei_classification.Classifiers import CLASSIFIERS


def generate_summurized_answer(question_query, question_type):
    answers_ids = get_stackoverflow_answer_ids(question_query, question_type)
    stacked_answers = get_answer_bodies(answers_ids)
    print(stacked_answers)
    summarized_answer = summarize(stacked_answers, word_count=50)
    if summarized_answer:
        return summarized_answer
    summarized_answer = summarize(stacked_answers)
    if summarized_answer == '':
        return "Sorry, I couldn't find any information in my datasources. :("

    return summarized_answer


def get_answer_bodies(answers_ids):
    answers = ''
    for answers_id in answers_ids:
        request_url = 'https://api.stackexchange.com/2.2/answers/' + \
        str(answers_id) + \
        '?order=desc&sort=votes&site=stackoverflow&filter=withbody'
    response = requests.get(request_url)
    answer = json.loads(response._content)['items'][0]['body']
    answer = remove_code_tag(answer)
    answer = html2text.html2text(answer)
    answer = ' '.join(re.split(r'\s{2,}|\n', answer)).strip()
    answers += '\n' + answer

    return answers


def get_stackoverflow_answer_ids(question_query, question_type):
    answers_ids = []
    print(question_query)
    request_url = "https://api.stackexchange.com/2.2/search/advanced?order=desc&sort=relevance&title=" +\
        question_query.lower() + "&site=stackoverflow"

    response = requests.get(request_url)

    questions = json.loads(
        response._content, object_pairs_hook=OrderedDict)['items']
    pprint(questions)
    for question in questions:
        print(question_type, question['title'], CLASSIFIERS.predict_question_type(
            question['title']))
        if question['is_answered'] is True and \
            'accepted_answer_id' in question.keys() and \
            (CLASSIFIERS.predict_question_type(question['title']) == question_type or
             title_contains(question['title'], question_query)):
            answers_ids.append(question['accepted_answer_id'])
        if len(answers_ids) >= 3:
            return answers_ids
    return answers_ids


def process_question(question):
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords += ['know', 'think', 'tell']
    rake_algo = Rake(stopwords=stopwords)
    rake_algo.extract_keywords_from_text(question)
    question_query_list = rake_algo.get_ranked_phrases()
    return (' '.join(question_query_list),
            CLASSIFIERS.predict_question_type(question))


def try_answer(response):
    fb_nlp_class = response['entry'][0]['messaging'][0]['message']['nlp']['entities']
    for k in fb_nlp_class.keys():
        if k == 'greetings':
           return check_confidence_and_return('Hi', fb_nlp_class[k])
        if k == 'bye':
            return check_confidence_and_return('Bye', fb_nlp_class[k])
        if k == 'thanks':
            return check_confidence_and_return('It was pleasure to help :-)',
                                               fb_nlp_class[k])

    message = response['entry'][0]['messaging'][0]['message']['text']
    question_query, question_type = process_question(message)
    return generate_summurized_answer(question_query, question_type)


def check_confidence_and_return(message, pred_dict):
     for el in pred_dict:
            if el['confidence'] > 0.9:
                return message


def title_contains(overflow_q, user_q):
    user_q_words = set(user_q.lower().split())
    overflow_q_words = set(overflow_q.lower().split())
    overlapped_words = user_q_words & overflow_q_words
    return 1 - ((len(user_q_words) - len(overflow_q_words)) / len(user_q_words)) > 0.7


def remove_code_tag(html):
    regex = r"<code>.*?</code>"
    matches = re.finditer(regex, html, re.DOTALL)
    for match in matches:
        html = html.replace(match.group(), '')
    return html
