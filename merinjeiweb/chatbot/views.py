from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic import View

from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import json
import requests
from pprint import pprint


# Create your views here.
class ChatBot(View):
    def get(self, request, *args, **kwargs):
        if self.request.GET['hub.verify_token'] == '19990402':
            print("IS THIS SOMTIAHSDKJHAGSJDKHgASJHDGAJhsgd")
            return HttpResponse(self.request.GET['hub.challenge'])
        else:
            return HttpResponse('Error, invalid token')
 
    @method_decorator(csrf_exempt)
    def dispatch(self, request, *args, **kwargs):

        return View.dispatch(self, request, *args, **kwargs)

    # Post function to handle Facebook messages
    def post(self, request, *args, **kwargs):
        # Converts the text payload into a python dictionary
        incoming_message = json.loads(self.request.body.decode('utf-8'))

        access_token = "EAAFiycyl6vIBAJeC4oTMOvHY8qLUImZBeZAG3NWZBoJAG50thvlXkT6d12ZBpZB4NhCT814t6kZAMmZCzWTyqcvmvK4XK84hOTKZCx6A4rKH9gdywLRKwWIWnS3IREGOCizxxCwvvi3cI0vKkMuvKUZARU4THYwZBv0mcdrZB0A8w8xxgZDZD"
        incoming_message = incoming_message['entry'][0]['messaging'][0]
        recipient = incoming_message['sender']['id']
        message = incoming_message['message']['text']

        data = {
            "messaging_type": "RESPONSE",
            "recipient": {"id": recipient},
            "message": {"text": message}
        }

        response = requests.post(
            "https://graph.facebook.com/v2.6/me/messages?access_token=" + access_token, json=data)

        return HttpResponse()
