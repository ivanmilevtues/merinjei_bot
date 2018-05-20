from django.conf.urls import url
from django.contrib import admin
from django.conf.urls import include

from dashboard.views import profile_handler, login, details_for_page


urlpatterns = [
    url(r'^profile/', profile_handler),
    url(r'^login/', login),
    url(r'^page_details/', details_for_page)
]
