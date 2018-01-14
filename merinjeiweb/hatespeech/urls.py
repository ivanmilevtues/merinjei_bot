from django.conf.urls import url
from django.contrib import admin
from django.conf.urls import include

from hatespeech.views import profile_handler

urlpatterns = [
    url(r'^profile/', profile_handler)
]
