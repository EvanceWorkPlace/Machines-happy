from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import CrashResult
from .serializers import CrashResultSerializer
from aviator import ai
import pandas as pd


def dashboard(request):
    return render(request, 'core/dashboard.html')


class CrashResultListCreate(APIView):
    def get(self, request):
        qs = CrashResult.objects.all().order_by('-timestamp')[:1000]
        serializer = CrashResultSerializer(qs, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = CrashResultSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class SuggestionView(APIView):
    def get(self, request):
        qs = CrashResult.objects.all().order_by('-timestamp')
        df = pd.DataFrame(list(qs.values('multiplier', 'timestamp')))
        suggestion = ai.generate_suggestion(df)
        summary = ai.summarize_results(df)
        return Response({'suggestion': suggestion, 'summary': summary})
