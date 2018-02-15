from django.conf.urls import url
from django.contrib import admin
from hatespeech.views import CommentScanner

urlpatterns = [
    url(r'^scan_for_hatespeech/', CommentScanner.scan_page),
    url(r'^subscribe', CommentScanner.subscribe),
    url(r'^unsubscribe', CommentScanner.unsubscribe)
]
