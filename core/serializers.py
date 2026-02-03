from rest_framework import serializers
from .models import CrashResult


class CrashResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = CrashResult
        fields = ['id', 'round_id', 'multiplier', 'timestamp']
