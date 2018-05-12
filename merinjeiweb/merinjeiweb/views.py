from django.http import HttpResponseRedirect
from django.views.generic import View
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

from chatbot.views import ChatBot
from hatespeech.views import CommentScanner
from dashboard.models import Page

import json
from pprint import pprint
import requests

def redirect_to_login(request):
    if request.user.is_authenticated():
        return HttpResponseRedirect('/logged/profile')
    return HttpResponseRedirect('/logged/login')


class WebHookHandler(View):
    def get(self, request, *args, **kwargs):
        if self.request.GET['hub.verify_token'] == '19990402':
            return HttpResponse(self.request.GET['hub.challenge'])
        return HttpResponse('Error, invalid token')

    @method_decorator(csrf_exempt)
    def dispatch(self, request, *args, **kwargs):
        return View.dispatch(self, request, *args, **kwargs)    

    def post(self, request, *args, **kwargs):
        parsed_request = json.loads(self.request.body.decode('utf-8'))['entry'][0]
        if 'messaging' in parsed_request and validate_chatbot_alive(parsed_request):
           ChatBot.process_messenger(request)
           return HttpResponse()
        if 'changes' in parsed_request.keys():
            changes = parsed_request['changes'][0]
            if 'field' in changes and changes['field'] == 'feed'\
                    and validate_feed_alive(parsed_request):
                CommentScanner.process_new_comment(request)
        return HttpResponse()



def validate_chatbot_alive(request):
    page_id = request['id']
    return Page.objects.filter(id=page_id)\
            .first().messenger_subscription


def validate_feed_alive(request):
    page_id = request['id']
    return Page.objects.filter(id=page_id)\
        .first().feed_subscription
