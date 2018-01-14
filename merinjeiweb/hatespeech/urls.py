from django.conf.urls import url
from django.contrib import admin
from hatespeech.views import profile_handler

urlpatterns = [
    url(r'^scan_for_hatespeech/', profile_handler)
]
