from django.http import HttpResponseRedirect
from django.views.generic import View
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

from chatbot.views import ChatBot
from hatespeech.views import CommentScanner

import json
from pprint import pprint
import requests

def redirect_to_login(request):
    if request.user.is_authenticated():
        return HttpResponseRedirect('/logged/profile')
    return HttpResponseRedirect('/logged/login')


class WebHookHandler(View):
    def get(self, request, *args, **kwargs):
        print('WebHookHandler was called')
        if self.request.GET['hub.verify_token'] == '19990402':
            return HttpResponse(self.request.GET['hub.challenge'])
        return HttpResponse('Error, invalid token')

    @method_decorator(csrf_exempt)
    def dispatch(self, request, *args, **kwargs):
        return View.dispatch(self, request, *args, **kwargs)    

    def post(self, request, *args, **kwargs):
        parsed_request = json.loads(self.request.body.decode('utf-8'))['entry'][0]
        pprint(parsed_request)
        if 'messaging' in parsed_request:
           ChatBot.process_messenger(request)
           return HttpResponse()
        parsed_request = parsed_request['changes'][0]
        if 'field' in parsed_request and parsed_request['field'] == 'feed':
            CommentScanner.process_new_comment(request)
        return HttpResponse()
