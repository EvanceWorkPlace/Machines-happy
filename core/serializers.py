# serializers.py
from rest_framework import serializers
from .models import CrashResult

class CrashResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = CrashResult
        fields = "__all__"
