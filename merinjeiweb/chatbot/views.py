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
from merinjei_classification.Classifiers import Classifiers

# Create your views here.
class ChatBot(View):
    def get(self, request, *args, **kwargs):
        if self.request.GET['hub.verify_token'] == '19990402':
            return HttpResponse(self.request.GET['hub.challenge'])
        else:
            return HttpResponse('Error, invalid token')
 
    @method_decorator(csrf_exempt)
    def dispatch(self, request, *args, **kwargs):

        return View.dispatch(self, request, *args, **kwargs)

    def post(self, request, *args, **kwargs):
        incoming_message = json.loads(self.request.body.decode('utf-8'))

        access_token = "EAAFiycyl6vIBAJeC4oTMOvHY8qLUImZBeZAG3NWZBoJAG50thvlXkT6d12ZBpZB4NhCT814t6kZAMmZCzWTyqcvmvK4XK84hOTKZCx6A4rKH9gdywLRKwWIWnS3IREGOCizxxCwvvi3cI0vKkMuvKUZARU4THYwZBv0mcdrZB0A8w8xxgZDZD"
        incoming_message = incoming_message['entry'][0]['messaging'][0]
        pprint(incoming_message)
        try:
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
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(e)
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


        return HttpResponse()


def process_message(message):
    answer = ""
    request_url = "https://api.stackexchange.com/2.2/search/advanced?order=desc&sort=activity&title=" + message + "&site=stackoverflow"

    response = requests.get(request_url)
    
    items = json.loads(response._content)['items']
    response_thread = None
    for thread in items:
        if thread['is_answered'] == True:
            response_thread = thread
            break

    answer_id = response_thread['accepted_answer_id']
    request_url = 'https://api.stackexchange.com/2.2/answers/' + str(answer_id) + '?order=desc&sort=activity&site=stackoverflow&filter=withbody'
    response = requests.get(request_url)
    answer = json.loads(response._content)['items'][0]['body']
    answer = html2text.html2text(answer)
    print("RAW ANSWER:\n|" + answer + "|")
    answer = '. '.join(re.split(r'\s{2,}', answer)).strip()
    print('AFTER SPLIT:\n|' + answer + '|')
    answer = summarize(answer)
    print('SUMMERIZED:\n|' + answer + '|')
    return answer
