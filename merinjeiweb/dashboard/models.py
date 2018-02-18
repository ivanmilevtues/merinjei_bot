from django.db import models


class PageSubscriptions(models.Model):
    id = models.CharField(max_length=255, primary_key=True)
    feed_subscription = models.BooleanField(default=False)
    messenger_subscription = models.BooleanField(default=False)
