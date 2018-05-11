import json
import requests
import numpy as np

from django.http import HttpResponse
from django.views.generic import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

from hatespeech.models import AccessTokens
from CONSTANTS import APP_ID, COMMENTS_CALLBACK, VERIFY_TOKEN, APP_SECRET
from allauth.socialaccount.models import SocialToken
from dashboard.models import PageSubscriptions
from pprint import pprint


from merinjei_classification.Classifiers import CLASSIFIERS


def get_page_posts(access_token, page_id):
    response = requests.get(
        'https://graph.facebook.com/v2.11/' + page_id + '/posts?access_token='
        + access_token)
    page_posts = json.loads(response._content)['data']
    return page_posts


def score_comments(comments):
    data = []
    comments_to_delete = []
    IS_HATE = 0
    hlp = CLASSIFIERS.get_hs_parser()
    for comment in comments:
        data.append(hlp.parse_line(comment['message'])[0])
    data = np.array(data)
    scored_comments = CLASSIFIERS.predict_parsed_comments(data)
    comments_to_delete = [comments[i] for i in range(len(scored_comments))\
                          if scored_comments[i] == IS_HATE]
    return comments_to_delete


def get_comments_for_post(posts, access_token):
    comments = []
    for post in posts:
        post_id = post['id']
        response = requests.get(
            'https://graph.facebook.com/v2.11/' + post_id +
            '/comments?access_token=' + access_token)
        data = json.loads(response._content)['data']
        comments += data
    return comments


def delete_comments(comments_to_del):
    for page_id, comments in comments_to_del.items():
        access_token = AccessTokens.objects.filter(id=page_id)
        if access_token.count() == 0:
            continue
        access_token = access_token.first().access_token
        for comment in comments:
            from pprint import pprint
            pprint(comment)
            try:
                response = requests.delete('https://graph.facebook.com/v2.11/' +
                                           comment['id'] + '?access_token=' +
                                           access_token)
            except KeyError:
                response = requests.delete('https://graph.facebook.com/v2.11/' +
                                           comment['comment_id'] +
                                           '?access_token=' +
                                           access_token)
            print(json.loads(response._content))


class CommentScanner(View):
    # This will scan the page at first.
    @staticmethod
    def scan_page(request):
        access_token = str(SocialToken.objects.get(
            account__user=request.user, account__provider='facebook'))
        page_id = request.POST.get('page_id')

        posts = get_page_posts(access_token, page_id)
        comments = get_comments_for_post(posts, access_token)
        comments_to_del = {}
        comments_to_del[page_id] = score_comments(comments)
        delete_comments(comments_to_del)
        return HttpResponse()

    # The purpose of this method is to recieve the subscribed webhooks
    # and call the needed handlers for certain messages
    @staticmethod
    def process_new_comment(request):
        incoming_message = json.loads(request.body.decode('utf-8'))
        from pprint import pprint
        pprint(incoming_message)
        entries = incoming_message['entry']
        messages = []
        comments_to_del = {}
        for entry in entries:
            changes = entry['changes']
            page_id = entry['id']
            messages = []
            for change in changes:
                if change['value']['verb'] == 'remove':
                    return HttpResponse()
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
        
        access_token = APP_ID + '|' + APP_SECRET
        data = {
            'object': 'page',
            'callback_url': COMMENTS_CALLBACK,
            'fields': ['feed'],
            'verify_token': VERIFY_TOKEN,
            'access_token': access_token,
        }
        response = requests.post('https://graph.facebook.com/v2.11/' +
                      page_id + '/subscriptions', data)
        pprint(json.loads(response._content))
        obj, _ = PageSubscriptions.objects.update_or_create(
            id=page_id,
            defaults={'feed_subscription': True})
        obj.save()

        return HttpResponse()

    @staticmethod
    def unsubscribe(request):
        page_id = request.POST.get('page_id')
        obj, _ = PageSubscriptions.objects.update_or_create(
            id=page_id,
            defaults={'feed_subscription': False}
        )
        obj.save()

        return HttpResponse()
