# -*- coding: utf-8 -*-
# Generated by Django 1.11.8 on 2018-05-11 16:38
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dashboard', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Page',
            fields=[
                ('id', models.CharField(max_length=256, primary_key=True, serialize=False)),
                ('feed_subscription', models.BooleanField(default=False)),
                ('messenger_subscription', models.BooleanField(default=False)),
                ('name', models.CharField(max_length=256)),
            ],
        ),
        migrations.DeleteModel(
            name='PageSubscriptions',
        ),
    ]
