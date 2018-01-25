import json
import requests

from django.http import HttpResponse
from django.views.generic import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

from hatespeech.models import AccessTokens
from CONSTANTS import APP_ID, COMMENTS_CALLBACK, VERIFY_TOKEN, APP_SECRET
from allauth.socialaccount.models import SocialToken


from merinjei_classification.Classifiers import CLASSIFIERS


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
            'https://graph.facebook.com/v2.11/' + post_id + '/comments?access_token=' +
            access_token)
        data = json.loads(response._content)['data']
        comments += data
    return comments


def delete_comments(comments_to_del):
    print(comments_to_del)
    for page_id, comments in comments_to_del.items():
        access_token = AccessTokens.objects.filter(id=page_id)
        if access_token.count() == 0:
            continue
        access_token = access_token.first().access_token
        for comment in comments:
            response = requests.delete('https://graph.facebook.com/v2.11/' +
                                       comment['comment_id'] + '?access_token=' + access_token)
            print(json.loads(response._content))


class CommentScanner(View):
    def get(self, request):
        if self.request.GET['hub.verify_token'] == '19990402':
            return HttpResponse(self.request.GET['hub.challenge'])
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
            access_token = [page_id]
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

        access_token = APP_ID + '|' + APP_SECRET
        data = {
            'object': 'page',
            'callback_url': COMMENTS_CALLBACK,
            'fields': ['feed'],
            'verify_token': VERIFY_TOKEN,
            'access_token': access_token,
            'active': True

        }
        requests.post('https://graph.facebook.com/v2.11/' +
                      page_id + '/subscriptions', data)
        return HttpResponse()
