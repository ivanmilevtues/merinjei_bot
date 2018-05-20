import json
import requests

from django.http import HttpResponseRedirect
from django.shortcuts import render

from allauth.socialaccount.models import SocialToken
from hatespeech.models import AccessTokens
from CONSTANTS import COMMENTS_CALLBACK, APP_ID, APP_SECRET
from pprint import pprint
from dashboard.models import Page
from hatespeech.models import DeletedPageComments


def profile_handler(request):
    if not request.user.is_authenticated():
        return HttpResponseRedirect('/logged/login');
    user = request.user
    access_token = SocialToken.objects.get(
        account__user=user, account__provider='facebook')
    access_token = access_token.token

    user_response = requests.get(
        'https://graph.facebook.com/v2.11/me?fields=picture,name&access_token=' +
        access_token)

    if user_response.status_code == 400:
        return HttpResponseRedirect('/logged/login?session_expired')

    fb_pages_response = requests.get(
        'https://graph.facebook.com/v2.11/me/accounts?type=page&access_token=' +
        access_token)

    user_data = json.loads(user_response._content)
    fb_page_data = json.loads(fb_pages_response._content)

    username = user_data['name']
    user_pic = user_data['picture']['data']['url']

    fb_pages = {}

    for page in fb_page_data['data']:
        if not has_perms(page):
            continue
        feed_subscription, messenger_subscription = get_subsriptions_for_page(
            page['id'], page['name'])

        fb_pages[page['name']] = {
            'id': page['id'],
            'access_token': page['access_token'],
            'feed_subscription': feed_subscription,
            'messenger_subscription': messenger_subscription
        }
        obj, _ = AccessTokens.objects.update_or_create(
            id=page['id'],
            defaults={'access_token': page['access_token']})

        obj.save()
    return render(request, 'dashboard.html', locals())


def login(request):
    if request.user.is_authenticated() and 'session_expired' not in request.GET.keys():
        return HttpResponseRedirect('/logged/profile')
    return render(request, 'login.html', locals())


def get_subsriptions_for_page(page_id, page_name):
    page_subscriptions = Page.objects.filter(id=page_id)
    if page_subscriptions.count() == 0:
        obj = Page.objects.create(id=page_id, name=page_name)
        page_subscriptions = obj
        obj.save()
    else:
        page_subscriptions = page_subscriptions.first()
    return (page_subscriptions.feed_subscription,
            page_subscriptions .messenger_subscription)

def has_perms(page):
    return 'ADMINISTER' in page['perms'] or 'EDIT_PROFILE' in page['perms'] \
        or 'MODERATE_CONTENT' in page['perms'] or 'BASIC_ADMIN' in page['perms']


def details_for_page(request):
    page_name =  request.GET.get('page_name')
    print(request.GET)
    page = Page.objects.filter(name=page_name).first()
    print(page)
    deleted_comments =  [c.deleted_comment for c in 
                        DeletedPageComments.objects.filter(page=page)]
    print(deleted_comments)
    return render(request, 'page_details.html', locals())
