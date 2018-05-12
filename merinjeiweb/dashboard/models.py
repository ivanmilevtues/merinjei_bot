from django.db import models


class Page(models.Model):
    id = models.CharField(max_length=256, primary_key=True)
    feed_subscription = models.BooleanField(default=False)
    messenger_subscription = models.BooleanField(default=False)
    name = models.CharField(max_length=256)
