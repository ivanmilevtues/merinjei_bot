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

    if user_response.status_code == 400:
        # should say session timeouted!
        return HttpResponseRedirect('/logged/login?session_expired')

    fb_pages_response = requests.get(
        'https://graph.facebook.com/v2.11/me/accounts?type=page&access_token=' + access_token)
    from pprint import pprint
    print(access_token)
    user_data = json.loads(user_response._content)
    fb_page_data = json.loads(fb_pages_response._content)
    pprint(user_data)
    pprint(fb_page_data)
    username = user_data['name']
    user_pic = user_data['picture']['data']['url']

    fb_pages = {}

    for page in fb_page_data['data']:
        fb_pages[page['name']] = {'id': page['id'], 'access_token': page['access_token']}
        obj, _ = AccessTokens.objects.update_or_create(
            id=page['id'],
            defaults={'access_token': page['access_token']})

        obj.save()
    print([(page.access_token, page.id) for page in list(AccessTokens.objects.all())])
    return render(request, 'profile.html', locals())


def login(request):
    if request.user.is_authenticated() and 'session_expired' not in request.GET.keys():
        return HttpResponseRedirect('/logged/profile')
    return render(request, 'login.html', locals())
