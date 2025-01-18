# Generated by Django 5.1.5 on 2025-01-18 06:50

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('api', '0001_initial'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.AddField(
            model_name='collaboration',
            name='collaborator',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
        ),
        migrations.AddField(
            model_name='project',
            name='collaborators',
            field=models.ManyToManyField(related_name='collaborators', through='api.Collaboration', to=settings.AUTH_USER_MODEL),
        ),
        migrations.AddField(
            model_name='collaboration',
            name='project',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='api.project'),
        ),
    ]
