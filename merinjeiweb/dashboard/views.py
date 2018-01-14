from django.shortcuts import render
from allauth.socialaccount.models import SocialToken, SocialApp
import requests
import json


def profile_handler(request):
    user = request.user
    access_token = SocialToken.objects.get(
        account__user=user, account__provider='facebook')
    access_token = access_token.token
    response = requests.get('https://graph.facebook.com/v2.11/me?access_token=' + access_token)
    data = json.loads(response._content)
    user = data['name']

    # This code will be enabled after facebook review for the application
    # user_id =  data['id']
    # response = requests.get('https://graph.facebook.com/v2.11/' + user_id + '/accounts?access_token=' + access_token)
    # pages_with_perms = json.loads(response._content)['data']

    return render(request, 'profile.html', locals())
