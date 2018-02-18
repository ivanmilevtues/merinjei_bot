import json
import requests

from django.http import HttpResponseRedirect
from django.shortcuts import render

from allauth.socialaccount.models import SocialToken
from hatespeech.models import AccessTokens
from CONSTANTS import COMMENTS_CALLBACK, APP_ID, APP_SECRET
from pprint import pprint
from dashboard.models import PageSubscriptions


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
        feed_subscription, messenger_subscription = get_subsriptions_for_page(
            page['id'])

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
    return render(request, 'profile.html', locals())


def login(request):
    if request.user.is_authenticated() and 'session_expired' not in request.GET.keys():
        return HttpResponseRedirect('/logged/profile')
    return render(request, 'login.html', locals())


def get_subsriptions_for_page(page_id):
    page_subscriptions = PageSubscriptions.objects.filter(id=page_id)
    if page_subscriptions.count() == 0:
        obj = PageSubscriptions.objects.create(id=page_id)
        page_subscriptions = obj
        obj.save()
    else:
        page_subscriptions = page_subscriptions.first()
    return (page_subscriptions.feed_subscription,
            page_subscriptions .messenger_subscription)
