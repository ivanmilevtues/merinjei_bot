from django.shortcuts import render
from allauth.socialaccount.models import SocialToken
from pprint import pprint
import requests
import json
from merinjei_classification.Classifiers import CLASSIFIERS
from django.http import HttpResponse
from django.views.generic import View

from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from CONSTANTS import APP_ID, COMMENTS_CALLBACK, VERIFY_TOKEN, APP_SECRET, DOMAIN, access_tokens

def get_page_posts(access_token, page_id):
    response = requests.get(
        'https://graph.facebook.com/v2.11/' + page_id + '/posts?access_token=' + access_token)
    page_posts = json.loads(response._content)['data']
    return page_posts


def score_comments(comments):
    comments_to_delete = []
    for comment in comments:
        if CLASSIFIERS.predict_comment_type(comment['message'])[0] == 0:
           comments_to_delete.append(comment)
    return comments_to_delete


def get_comments_for_post(posts, access_token):
    comments = []
    for post in posts:
        post_id = post['id']
        response = requests.get(
            'https://graph.facebook.com/v2.11/' + post_id + '/comments?access_token=' + access_token)
        data = json.loads(response._content)['data']
        comments += data
    return comments


def delete_comments(comments_to_del):
    print(comments_to_del)
    for page_id, comments in comments_to_del.items():
        print(page_id)
        print(access_tokens)
        if page_id in access_tokens.keys():
            access_token = access_tokens[page_id]
            for comment in comments:
                response = requests.delete('https://graph.facebook.com/v2.11/' + 
                    comment['comment_id'] + '?access_token=' + access_token)
                print(json.loads(response._content))
                    



class CommentScanner(View):
    def get(self, request):
        if self.request.GET['hub.verify_token'] == '19990402':
            return HttpResponse(self.request.GET['hub.challenge'])
        else:
            return HttpResponse('Error, invalid token')

    @method_decorator(csrf_exempt)
    def dispatch(self, request, *args, **kwargs):
        return View.dispatch(self, request, *args, **kwargs)

    # This will scan the page at first.
    @staticmethod
    def scan_page(request):
        access_token = SocialToken.objects.get(
            account__user=request.user, account__provider='facebook')
        access_token = access_token.token
        page_id = request.POST.get('page_id')

        posts = get_page_posts(access_token, page_id)
        comments = get_comments_for_post(posts, access_token)
        comments_to_del = {}
        comments_to_del[page_id] = score_comments(comments)

        for page_id, comments in comments_to_del.items():
            print(page_id, access_tokens)
            if page_id in access_tokens.keys():
                access_token = access_tokens[page_id]
                for comment in comments:
                    response = requests.delete('https://graph.facebook.com/v2.11/' + 
                        comment['id'] + '?access_token=' + access_token)
                    print(json.loads(response._content))
        return HttpResponse()
        

    # The purpose of this method is to recieve the subscribed webhooks
    # and call the needed handlers for certain messages
    def post(self, request):
        incoming_message = json.loads(self.request.body.decode('utf-8'))
        entries = incoming_message['entry']
        messages = []
        comments_to_del = {}
        for entry in entries:
            page_id = entry['id']
            changes = entry['changes']
            messages = []
            for change in changes:
                if change['field'] == 'feed':
                    messages.append(change['value'])
            if page_id not in comments_to_del.keys():
                comments_to_del[page_id] = []
            comments_to_del[page_id] += score_comments(messages)
        delete_comments(comments_to_del)
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
            'callback_url': COMMENTS_CALLBACK,
            'fields': ['feed'],
            'verify_token': VERIFY_TOKEN,
            'access_token': access_token,
            'active': True

        }
        response = requests.post('https://graph.facebook.com/v2.11/' +
                                 page_id + '/subscriptions', data)
        return HttpResponse()
