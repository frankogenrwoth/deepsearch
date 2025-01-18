from django.db import models

from django.contrib.auth import get_user_model


UserModel = get_user_model()

# Create your models here.
class Project(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()

    is_active = models.BooleanField(default=True)
    date_created = models.DateTimeField(auto_now_add=True)
    last_modified = models.DateTimeField(auto_now=True)

    collaborators = models.ManyToManyField(UserModel, related_name='collaborators', through='Collaboration')

    def __str__(self):
        return self.name
    
class Collaboration(models.Model):
    class CollaboratorRole(models.TextChoices):
        OWNER = 'OWNER'
        ADMIN = 'ADMIN'
        MEMBER = 'MEMBER'

    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    collaborator = models.ForeignKey(UserModel, on_delete=models.CASCADE)
    role = models.CharField(max_length=100)

    def __str__(self):
        return f'{self.collaborator.username} - {self.project.name}'