import json
import requests
import numpy as np
from pprint import pprint
from dashboard.models import Page
from hatespeech.models import AccessTokens

from merinjei_classification.Classifiers import CLASSIFIERS

from hatespeech.models import DeletedPageComments


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
    comments_to_delete = [comments[i] for i in range(len(scored_comments))
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


def save_comment(comment, page_id):
    page = Page.objects.filter(id=page_id).first()
    db_instance = DeletedPageComments(page=page,
                                      deleted_comment=comment['message'])
    db_instance.save()


def delete_comments(comments_to_del):
    for page_id, comments in comments_to_del.items():
        access_token = AccessTokens.objects.filter(id=page_id)
        if access_token.count() == 0:
            continue
        access_token = access_token.first().access_token
        for comment in comments:
            save_comment(comment, page_id)
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
