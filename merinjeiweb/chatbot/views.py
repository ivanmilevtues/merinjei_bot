from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic import View

from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

from pprint import pprint
import json
import nltk
import html2text
import numpy as np
import requests
import re

from hatespeech.models import AccessTokens
from gensim.summarization import summarize
from merinjei_classification.Classifiers import CLASSIFIERS
from CONSTANTS import APP_ID, VERIFY_TOKEN, APP_SECRET, DOMAIN, MESSENGER_CALLBACK

class ChatBot(View):
    def get(self, request, *args, **kwargs):
        if self.request.GET['hub.verify_token'] == VERIFY_TOKEN:
            return HttpResponse(self.request.GET['hub.challenge'])
        return HttpResponse('Error, invalid token')
 
    @method_decorator(csrf_exempt)
    def dispatch(self, request, *args, **kwargs):

        return View.dispatch(self, request, *args, **kwargs)

    # The purpose of this method is to recieve the subscribed webhooks
    # and call the needed handlers for certain messages
    def post(self, request, *args, **kwargs):
        response = json.loads(self.request.body.decode('utf-8'))

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
            'active': True

        }

        nlp_data = {
            'access_token': page_access_token
        }
        
        response = requests.post(
            'https://graph.facebook.com/v2.11/me/nlp_configs?nlp_enabled=true', nlp_data)
        pprint(json.loads(response._content))

        response = requests.post('https://graph.facebook.com/v2.11/' +
                                 page_id + '/subscriptions', data)
        pprint(json.loads(response._content))
        return HttpResponse()


def generate_summurized_answer(message):
    answer = "Some answer?"
    # request_url = "https://api.stackexchange.com/2.2/search/advanced?order=desc&sort=votes&title=" +\
    #                message + "&site=stackoverflow"

    # response = requests.get(request_url)
    # pprint(json.loads(response._content))
    # items = json.loads(response._content)['items']
    # answers = ''
    # pprint(items)
    # for thread in items:
    #     if thread['is_answered'] is True and 'accepted_answer_id' in thread.keys():
    #         answer_id = thread['accepted_answer_id']
    #         break

    # request_url = 'https://api.stackexchange.com/2.2/answers/' + str(answer_id) +\
    #             '?order=desc&sort=votes&site=stackoverflow&filter=withbody'
    # response = requests.get(request_url)
    # answer = json.loads(response._content)['items'][0]['body']
    # answer = html2text.html2text(answer)
    # answer = ' '.join(re.split(r'\s{2,}|\n', answer)).strip()
    # answers += '\n' + answer

    # print(answers)
    # summerized_answer = summarize(answers, word_count=50)
    # print('\n\n\nSUMMERIZED:\n|' + summerized_answer + '|')
    # if summerized_answer:
    #     return summerized_answer
    return answer


def process_question(question):
    proba = CLASSIFIERS.predict_proba_question_type(question)
    if np.count_nonzero(proba > 0.3) == 0:
        raise Exception('Not a question')
    question_words = nltk.word_tokenize(question)
    pos_tagged = nltk.pos_tag(question_words)
    print(pos_tagged)
    for w, t in pos_tagged:
        if t not in ['WP', 'VBZ', 'DT']:
            yield w


def try_answer(response):
    fb_nlp_class = response['entry'][0]['messaging'][0]['message']['nlp']['entities']
    for k in fb_nlp_class.keys():
        if k == 'greetings':
           return check_confidence_and_return('Hi', fb_nlp_class[k])
        if k == 'bye':
            return check_confidence_and_return('Bye', fb_nlp_class[k])
    
    message = response['entry'][0]['messaging'][0]['message']['text']
    message_query = process_question(message)
    return generate_summurized_answer(message_query)


def check_confidence_and_return(message, pred_dict):
     for el in pred_dict:
            if el['confidence'] > 0.9:
                return message
