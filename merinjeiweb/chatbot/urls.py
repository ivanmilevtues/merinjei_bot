from django.conf.urls import url
from django.contrib import admin
from chatbot.views import ChatBot, backup_chatbot

urlpatterns = [
    url(r'^subscribe', ChatBot.subscribe),
    url(r'^unsubscribe', ChatBot.unsubscribe),
    url(r'^backup', backup_chatbot)
]
