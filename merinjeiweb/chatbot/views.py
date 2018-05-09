from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic import View

from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

import json
import nltk
import html2text
import numpy as np
import requests
import re
from pprint import pprint
from rake_nltk import Rake
from collections import OrderedDict

from dashboard.models import PageSubscriptions
from hatespeech.models import AccessTokens
from gensim.summarization import summarize
from merinjei_classification.Classifiers import CLASSIFIERS
from CONSTANTS import APP_ID, VERIFY_TOKEN, APP_SECRET, DOMAIN, MESSENGER_CALLBACK

class ChatBot(View):
    @method_decorator(csrf_exempt)
    def dispatch(self, request, *args, **kwargs):

        return View.dispatch(self, request, *args, **kwargs)

    # The purpose of this method is to recieve the subscribed webhooks
    # and call the needed handlers for certain messages
    @staticmethod
    def process_messenger(request):
        response = json.loads(request.body.decode('utf-8'))

        if 'delivery'in response['entry'][0]['messaging'][0].keys():
            return HttpResponse()
        if 'message' not in response['entry'][0]['messaging'][0].keys():
            return HttpResponse()
        if 'is_echo' in response['entry'][0]['messaging'][0]['message'].keys():
            return HttpResponse()


        answer = ""
        try:
            answer = try_answer(response)
        except Exception as e:
            pprint(e)
            answer = "I couldn't understand you may you try once again with other words."

        # Send a resposponse
        page_id = response['entry'][0]['id']
        recipient = response['entry'][0]['messaging'][0]['sender']['id']
        access_token = AccessTokens.objects.filter(id=page_id).first().access_token
        data = {
            "messaging_type": "RESPONSE",
            "recipient": {"id": recipient},
            "message": {"text": answer}
        }

        response = requests.post(
            "https://graph.facebook.com/v2.6/me/messages?access_token=" + access_token,
            json=data)
        print(json.loads(response._content))
        return HttpResponse(200)

    @staticmethod
    def subscribe(request):
        
        page_id = request.POST.get('page_id')
        page_access_token = request.POST.get('access_token')
        access_token = APP_ID + '|' + APP_SECRET
        data = {
            'object': 'page',
            'callback_url': MESSENGER_CALLBACK,
            'fields': ['messages'],
            'verify_token': VERIFY_TOKEN,
            'access_token': access_token,
        }

        nlp_data = {
            'access_token': page_access_token
        }
        
        app_data = {
            'access_token': page_access_token
        }

        response = requests.post(
            "https://graph.facebook.com/v2.11/" +
            page_id + "/subscribed_apps", app_data)
        response = requests.post(
            'https://graph.facebook.com/v2.11/me/nlp_configs?nlp_enabled=true', nlp_data)
        response = requests.post('https://graph.facebook.com/v2.11/' +
                                 page_id + '/subscriptions', data)
        print("MESSENGER SUBSCRIPTION")                                 
        pprint(json.loads(response._content))
        obj, _ = PageSubscriptions.objects.update_or_create(
            id=page_id,
            defaults={'messenger_subscription': True})
        obj.save()
    
        return HttpResponse()

    @staticmethod
    def unsubscribe(request):
        page_id = request.POST.get('page_id')
        obj, _ = PageSubscriptions.objects.update_or_create(
            id=page_id,
            defaults={'messenger_subscription': False}
        )
        obj.save()

        return HttpResponse()

def generate_summurized_answer(question_query, question_type):
    answers = ""
    answers_ids = []
    print(question_query)
    request_url = "https://api.stackexchange.com/2.2/search/advanced?order=desc&sort=relevance&title=" +\
        question_query.lower() + "&site=stackoverflow"

    response = requests.get(request_url)

    questions = json.loads(response._content, object_pairs_hook=OrderedDict)['items']
    pprint(questions)
    for question in questions:
        print(question_type, question['title'], CLASSIFIERS.predict_question_type(
            question['title']))
        if question['is_answered'] is True and \
            'accepted_answer_id' in question.keys() and \
            (CLASSIFIERS.predict_question_type(question['title']) == question_type or\
            title_contains(question['title'], question_query)):
            answers_ids.append(question['accepted_answer_id'])
        if len(answers_ids) >= 3:
            break
    print(answers_ids)
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

    print(answers)
    summarized_answer = summarize(answers, word_count=50)
    if summarized_answer:
        return summarized_answer
    summarized_answer = summarize(answers)
    if summarized_answer == '':
        return "Sorry, I couldn't find any information in my datasources. :("

    return summarized_answer
    


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
