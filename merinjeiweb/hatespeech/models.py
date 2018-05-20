from django.db import models
from dashboard.models import Page


class AccessTokens(models.Model):
    id = models.CharField(max_length=255, primary_key=True)
    access_token = models.CharField(max_length=255)
    

class DeletedPageComments(models.Model):
    id = models.CharField(max_length=255, primary_key=True)
    deleted_comment = models.CharField(max_length=1024)
    page = models.ForeignKey(Page, on_delete=models.CASCADE)