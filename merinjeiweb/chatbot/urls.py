from django.conf.urls import url
from django.contrib import admin
from chatbot.views import ChatBot

urlpatterns = [
    url(r'^handlemessage/65cd8fe51586d2e451ae83320e7bd549d6841f46c2d27a870b', ChatBot.as_view())
]
