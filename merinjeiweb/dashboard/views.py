from django.shortcuts import render
from allauth.socialaccount.models import SocialToken, SocialApp
import requests
from pprint import pprint
from hatespeech.models import AccessTokens
import json


def profile_handler(request):
    user = request.user
    access_token = SocialToken.objects.get(
        account__user=user, account__provider='facebook')
    access_token = access_token.token
    print(request.user)
    print(request.user.__dict__)
    user_response = requests.get(
        'https://graph.facebook.com/v2.11/me?fields=picture,name&access_token=' + access_token)

    fb_pages_response = requests.get(
        'https://graph.facebook.com/v2.11/me/accounts?type=page&access_token=' + access_token)

    user_data = json.loads(user_response._content)
    fb_page_data = json.loads(fb_pages_response._content)
    username = user_data['name']
    user_pic = user_data['picture']['data']['url']

    fb_pages = {}

    for page in fb_page_data['data']:
        fb_pages[page['name']] = {'id': page['id'], 'access_token': page['access_token']}
        at = AccessTokens.objects.update_or_create(id = page['id'], access_token = page['access_token'])
        at.save()
    pprint(fb_pages)
    return render(request, 'profile.html', locals())

