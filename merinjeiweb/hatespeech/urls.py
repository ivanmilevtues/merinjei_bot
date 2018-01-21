from django.conf.urls import url
from django.contrib import admin
from hatespeech.views import CommentScanner

urlpatterns = [
    url(r'^scan_for_hatespeech/', CommentScanner.scan_page),
    url(r'^handle_comments/715aec20c75a416ca2385210013c9cb3',
        CommentScanner.as_view()),
    url(r'^subscribe', CommentScanner.subscribe)
]
