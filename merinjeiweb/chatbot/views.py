from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic import View

from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from pprint import pprint
import json
import html2text
import requests
import re

from gensim.summarization import summarize
from merinjei_classification.Classifiers import CLASSIFIERS
from CONSTANTS import APP_ID, VERIFY_TOKEN, APP_SECRET, DOMAIN, MESSENGER_CALLBACK

class ChatBot(View):
    def get(self, request, *args, **kwargs):
        if self.request.GET['hub.verify_token'] == '19990402':
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
            pprint(incoming_message)
            if 'is_echo' in incoming_message['message'].keys():
                return HttpResponse()

            recipient = incoming_message['sender']['id']
            message = incoming_message['message']['text']
            answer = process_message(message)

            data = {
                "messaging_type": "RESPONSE",
                "recipient": {"id": recipient},
                "message": {"text": answer}
            }

            response = requests.post(
                "https://graph.facebook.com/v2.6/me/messages?access_token=" + access_token, json=data)
        except Exception as e:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(e)
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        return HttpResponse()

    @staticmethod
    def subscribe(request):
        page_id = request.POST.get('page_id')
        # GET THE APP ACCESS_TOKEN
        response = requests.get(
            'https://graph.facebook.com/oauth/access_token?client_id=' +
            APP_ID + '&client_secret=' + APP_SECRET +
            '&grant_type=client_credentials')
        response1 = requests.get(
            'https://graph.facebook.com/endpoint?key=value&access_token={}|{}'.format(APP_ID, APP_SECRET))
        # Take the access token from the json result
        access_token_page = request.POST.get('access_token')
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

def process_message(message):
    answer = ""
    request_url = "https://api.stackexchange.com/2.2/search/advanced?order=desc&sort=votes&title=" + message + "&site=stackoverflow"

    response = requests.get(request_url)
    
    items = json.loads(response._content)['items']
    response_thread = None
    pprint(items)
    for thread in items:
        if thread['is_answered'] == True:
            response_thread = thread
            break

    answer_id = response_thread['accepted_answer_id']
    request_url = 'https://api.stackexchange.com/2.2/answers/' + str(answer_id) + '?order=desc&sort=votes&site=stackoverflow&filter=withbody'
    response = requests.get(request_url)
    answer = json.loads(response._content)['items'][0]['body']
    answer = html2text.html2text(answer)
    answer = ' '.join(re.split(r'\s{2,}|\n', answer)).strip()
    summerized_answer = summarize(answer, ratio=0.5)
    print('SUMMERIZED:\n|' + summerized_answer + '|')
    if summerized_answer:
        return summerized_answer
    else:
        return answer
