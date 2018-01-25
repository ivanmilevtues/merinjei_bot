from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic import View

from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from pprint import pprint
import json


import nltk
import html2text
import requests
import re
import numpy as np
from gensim.summarization import summarize
from merinjei_classification.Classifiers import CLASSIFIERS
from CONSTANTS import APP_ID, VERIFY_TOKEN, APP_SECRET, DOMAIN, MESSENGER_CALLBACK

class ChatBot(View):
    def get(self, request, *args, **kwargs):
        if self.request.GET['hub.verify_token'] == VERIFY_TOKEN:
            return HttpResponse(self.request.GET['hub.challenge'])
        else:
            return HttpResponse('Error, invalid token')
 
    @method_decorator(csrf_exempt)
    def dispatch(self, request, *args, **kwargs):

        return View.dispatch(self, request, *args, **kwargs)

    # The purpose of this method is to recieve the subscribed webhooks
    # and call the needed handlers for certain messages
    def post(self, request, *args, **kwargs):
        incoming_message = json.loads(self.request.body.decode('utf-8'))
        pprint(incoming_message)
        access_token = "EAAFiycyl6vIBAJeC4oTMOvHY8qLUImZBeZAG3NWZBoJAG50thvlXkT6d12ZBpZB4NhCT814t6kZAMmZCzWTyqcvmvK4XK84hOTKZCx6A4rKH9gdywLRKwWIWnS3IREGOCizxxCwvvi3cI0vKkMuvKUZARU4THYwZBv0mcdrZB0A8w8xxgZDZD"
        try:
            incoming_message = incoming_message['entry'][0]['messaging'][0]
            if 'is_echo' in incoming_message['message'].keys():
                return HttpResponse()

            recipient = incoming_message['sender']['id']
            message = incoming_message['message']['text']
            question_qeury = process_question(message)
            question_qeury = ' '.join(question_qeury)
            print(question_qeury)
            answer = generate_answer(question_qeury)

            data = {
                "messaging_type": "RESPONSE",
                "recipient": {"id": recipient},
                "message": {"text": answer}
            }

            response = requests.post(
                "https://graph.facebook.com/v2.6/me/messages?access_token=" + access_token, json=data)
        except Exception as e:
            print(e)
        return HttpResponse()

    @staticmethod
    def subscribe(request):
        page_id = request.POST.get('page_id')
        access_token = APP_ID + '|' + APP_SECRET
        data = {
            'object': 'page',
            'callback_url': MESSENGER_CALLBACK,
            'fields': ['messages'],
            'verify_token': VERIFY_TOKEN,
            'access_token': access_token,
            'active': True

        }
        response = requests.post('https://graph.facebook.com/v2.11/' +
                                 page_id + '/subscriptions', data)
        pprint(json.loads(response._content))
        return HttpResponse()


def generate_answer(message):
    answer = ""
    request_url = "https://api.stackexchange.com/2.2/search/advanced?order=desc&sort=votes&title=" +\
                   message + "&site=stackoverflow"

    response = requests.get(request_url)
    pprint(json.loads(response._content))
    items = json.loads(response._content)['items']
    answers = ''
    pprint(items)
    for thread in items:
        if thread['is_answered'] == True and 'accepted_answer_id' in thread.keys():
            answer_id = thread['accepted_answer_id']

        request_url = 'https://api.stackexchange.com/2.2/answers/' + str(answer_id) +\
                    '?order=desc&sort=votes&site=stackoverflow&filter=withbody'
        response = requests.get(request_url)
        answer = json.loads(response._content)['items'][0]['body']
        answer = html2text.html2text(answer)
        answer = ' '.join(re.split(r'\s{2,}|\n', answer)).strip()
        answers += '\n' + answer

    print(answers)
    summerized_answer = summarize(answers, word_count=50)
    print('\n\n\nSUMMERIZED:\n|' + summerized_answer + '|')
    if summerized_answer:
        return summerized_answer
    else:
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
