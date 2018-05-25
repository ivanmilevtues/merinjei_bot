from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic import View

from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

import json
import requests
from pprint import pprint

from dashboard.models import Page
from hatespeech.models import AccessTokens
from chatbot.generate_answer import try_answer
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
            answer = "Sorry, I couldn't understand :(. Would you try once again with other words."

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
        obj, _ = Page.objects.update_or_create(
            id=page_id,
            defaults={'messenger_subscription': True})
        obj.save()
    
        return HttpResponse()

    @staticmethod
    def unsubscribe(request):
        page_id = request.POST.get('page_id')
        obj, _ = Page.objects.update_or_create(
            id=page_id,
            defaults={'messenger_subscription': False}
        )
        obj.save()

        return HttpResponse()
