# views.py
from django.shortcuts import render
from rest_framework import generics
from rest_framework.views import APIView
from rest_framework.response import Response
from .utils import analyze_sequence, detect_streaks
from .models import CrashResult
from .serializers import CrashResultSerializer
from .utils import analyze_sequence
import statistics
import csv
from django.http import HttpResponse


def dashboard(request):
    return render(request, "core/dashboard.html")


class CrashResultListCreate(generics.ListCreateAPIView):
    queryset = CrashResult.objects.all()
    serializer_class = CrashResultSerializer

    def perform_create(self, serializer):
        instance = serializer.save()
        values = list(
            CrashResult.objects.values_list("multiplier", flat=True)
        )
        forecast, vol = analyze_sequence(values)
        instance.volatility = vol
        instance.save()



class SuggestionView(APIView):
    def get(self, request):
        values = list(
            CrashResult.objects.values_list("multiplier", flat=True)
        )

        forecast, vol = analyze_sequence(values)
        streaks = detect_streaks(values)

        return Response({
            "forecast": forecast,
            "volatility": vol,
            "streaks": streaks,
            "note": "Statistical sequence analysis (educational)"
        })


def export_csv(request):
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="aviator_sequence.csv"'

    writer = csv.writer(response)
    writer.writerow(["round_id", "multiplier", "timestamp", "volatility"])

    for r in CrashResult.objects.all():
        writer.writerow([
            r.round_id,
            r.multiplier,
            r.timestamp,
            r.volatility
        ])

    return response



class VolatilityHeatmapView(APIView):
    def get(self, request):
        values = list(
            CrashResult.objects.values_list("multiplier", flat=True)
        )

        window = 10
        heatmap = []

        for i in range(len(values)):
            if i < window - 1:
                heatmap.append(None)
                continue

            slice_ = values[i - window + 1:i + 1]
            std = statistics.stdev(slice_) if len(slice_) > 1 else 0
            heatmap.append(round(std, 2))

        return Response({
            "window": window,
            "volatility": heatmap
        })
