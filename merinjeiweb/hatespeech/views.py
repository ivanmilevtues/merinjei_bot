import json
import requests
import threading
from time import sleep
from pprint import pprint

from django.http import HttpResponse, JsonResponse
from django.views.generic import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

from merinjei_classification.Classifiers import CLASSIFIERS
<<<<<<< HEAD

=======
>>>>>>> master
from hatespeech.page_crawler import get_page_posts, get_comments_for_post,\
                                    score_comments, delete_comments
from CONSTANTS import APP_ID, COMMENTS_CALLBACK, VERIFY_TOKEN, APP_SECRET
from allauth.socialaccount.models import SocialToken
from dashboard.models import Page


class CommentScanner:
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
        print('COMMENTS WEBHOOK SUBSCRIBE')
        pprint(json.loads(response._content))
        obj, _ = Page.objects.update_or_create(
            id=page_id,
            defaults={'feed_subscription': True})
        obj.save()

        return HttpResponse()

    @staticmethod
    def unsubscribe(request):
        page_id = request.POST.get('page_id')
        obj, _ = Page.objects.update_or_create(
            id=page_id,
            defaults={'feed_subscription': False}
        )
        obj.save()

        return HttpResponse()
    

    @staticmethod
    def subscribe_polling(request):
        page_id = request.POST.get('page_id')
        if not Page.objects.filter(id=page_id)\
            .first().feed_subscription:
            polling_thread = CommentPollingThread(request)
            
            obj, _ = Page.objects.update_or_create(
                id=page_id,
                defaults={'feed_subscription': True})
            obj.save()
            
            polling_thread.start()
            
            return HttpResponse(status=200)
        return HttpResponse(status=500)


class CommentPollingThread(threading.Thread):
    def __init__(self, request):
        super(CommentPollingThread, self).__init__()
        self.request = request
    

    def run(self):
        page_id = self.request.POST.get('page_id')
        minutes = float(self.request.POST.get('minutes'))
        while True:
            print("Thread goes")
            print("minutes", minutes * 60)
            sleep(minutes * 60)
            if Page.objects.filter(id=page_id)\
                .first().feed_subscription:
                CommentScanner.scan_page(self.request)
            else:
                break

def backup_hatespeech_detect(request):
    msg = request.GET.get('message')
    pred = CLASSIFIERS.predict_comment_type(msg)
    return JsonResponse({'is_not_hatespeech': pred})