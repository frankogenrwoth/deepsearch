from rest_framework import serializers
from api.models import Collaborator

class CollaboratorSerializer(serializers.ModelSerializer):
    class Meta:
        model = Collaborator
        fields = '__all__'




