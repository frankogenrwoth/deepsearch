from rest_framework import serializers
from api.models import Collaborator, Project

class CollaboratorSerializer(serializers.ModelSerializer):
    class Meta:
        model = Collaborator
        fields = '__all__'


class ProjectSerializer(serializers.ModelSerializer):
    collaborators = CollaboratorSerializer(many=True, read_only=True)

    class Meta:
        model = Project
        fields = '__all__'
        exclude = ('embedding',)


class SearchSerializer(serializers.Serializer):
    match_id = serializers.IntegerField()
    data = serializers.CharField()
    key = serializers.CharField()
    score = serializers.FloatField()
    date_added = serializers.DateTimeField()