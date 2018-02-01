import json
import requests

from django.http import HttpResponseRedirect
from django.shortcuts import render

from allauth.socialaccount.models import SocialToken
from hatespeech.models import AccessTokens


def profile_handler(request):
    if not request.user.is_authenticated():
        return HttpResponseRedirect('/logged/login');
    user = request.user
    access_token = SocialToken.objects.get(
        account__user=user, account__provider='facebook')
    access_token = access_token.token

    user_response = requests.get(
        'https://graph.facebook.com/v2.11/me?fields=picture,name&access_token=' + access_token)

    fb_pages_response = requests.get(
        'https://graph.facebook.com/v2.11/me/accounts?type=page&access_token=' + access_token)

    user_data = json.loads(user_response._content)
    fb_page_data = json.loads(fb_pages_response._content)
    from pprint import pprint
    username = user_data['name']
    user_pic = user_data['picture']['data']['url']

    fb_pages = {}

    for page in fb_page_data['data']:
        fb_pages[page['name']] = {'id': page['id'], 'access_token': page['access_token']}
        obj, _ = AccessTokens.objects.update_or_create(
            id=page['id'],
            defaults={'access_token': page['access_token']})

        obj.save()
    return render(request, 'profile.html', locals())


def login(request):
    if request.user.is_authenticated():
        return HttpResponseRedirect('/logged/profile')
    return render(request, 'login.html', locals())
