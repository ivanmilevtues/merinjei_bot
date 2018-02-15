from django.conf.urls import url
from django.contrib import admin
from chatbot.views import ChatBot

urlpatterns = [
    url(r'^subscribe', ChatBot.subscribe),
    url(r'^unsubscribe', ChatBot.unsubscribe)
]
