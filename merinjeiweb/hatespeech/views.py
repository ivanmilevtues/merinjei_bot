from django.shortcuts import render
from allauth.socialaccount.models import SocialToken
from pprint import pprint
import requests
import json
from merinjei_classification.Classifiers import Classifiers
# Create your views here.


def get_page_posts(access_token):
    response = requests.get(
        'https://graph.facebook.com/v2.11/EuropeanCommission/posts?access_token=' + access_token)
    from pprint import pprint
    page_posts = json.loads(response._content)['data']
    return page_posts


def get_comments_for_post(posts, access_token):
    comments = []
    for post in posts:
        post_id = post['id']
        response = requests.get(
            'https://graph.facebook.com/v2.11/' + post_id + '/comments?access_token=' + access_token)
        data = json.loads(response._content)['data']
        comments += data
    return comments


def score_comments(comments, access_token):
    clf = Classifiers("merinjei_classification/classifiers/hatespeech_clf.pkl",
                      "merinjei_classification/data/features/hatespeech_features.pkl",
                      "merinjei_classification/classifiers/question_clf.pkl",
                      "merinjei_classification/data/features/questions_full_features.pkl")
    for comment in comments:
        if clf.predict_comment_type(comment['message'])[0] == 0: 
           print('This should be deleted')


def profile_handler(request):
    access_token = SocialToken.objects.get(
          account__user=request.user, account__provider='facebook')
    access_token = access_token.token
    posts = get_page_posts(access_token)
    comments = get_comments_for_post(posts, access_token)
    score_comments(comments, access_token)
