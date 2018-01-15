from django.conf.urls import url
from django.contrib import admin
from hatespeech.views import scan_page

urlpatterns = [
    url(r'^scan_for_hatespeech/', scan_page)
]
