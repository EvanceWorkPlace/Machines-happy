# views.py
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse, HttpRequest
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import generics
from .models import CrashResult, PredictionCheck
from .serializers import CrashResultSerializer
from core.ml_models import RandomForestPredictor
import csv
from rest_framework import status
from core.ml_models import AviatorPredictionService


import statistics
from django.db.models import Count



PREDICTION_TOLERANCE = 0.30

predictor = RandomForestPredictor(window_size=10)

def dashboard(request):
    return render(request, "core/dashboard.html")


class CrashResultListCreate(generics.ListCreateAPIView):
    queryset = CrashResult.objects.all()
    serializer_class = CrashResultSerializer

class StatsAPIView(APIView):
    def my_view(request):
        if request.method == 'GET':
            total = PredictionCheck.objects.count()
            passed = PredictionCheck.objects.filter(status="PASS").count()

            accuracy = round((passed / total) * 100, 2) if total > 0 else 0

            # streak (latest results)
            streak = 0
            last_status = None
            for r in PredictionCheck.objects.order_by('-created_at')[:20]:
                if last_status is None:
                    last_status = r.status
                    streak = 1
                elif r.status == last_status:
                    streak += 1
                else:
                    break

            return Response({
                "accuracy": accuracy,
                "streak": streak,
                "streak_type": last_status
            })
        else:
            pass



class PredictAPIView(APIView):
    def post(self, request):
        try:
            # ✅ Frontend sends a number, not an object
            current_value = float(request.data.get("currentValue"))

        except (TypeError, ValueError):
            return Response(
                {"error": "Invalid currentValue"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Historical values (training data you entered)
        values = list(
            CrashResult.objects
            .order_by("timestamp")
            .values_list("multiplier", flat=True)
        )

        if len(values) < 3:
            return Response({
                "prediction": None,
                "confidence": 0,
                "volatility": None,
                "risk": "UNKNOWN",
                "message": "Not enough data yet"
            })

        service = AviatorPredictionService()
        prediction = service.predict_next(values + [current_value])

        volatility = round(statistics.stdev(values[-10:]), 2) if len(values) >= 10 else 0

        # Simple risk bands
        if volatility < 0.5:
            risk = "LOW"
        elif volatility < 1.5:
            risk = "MID"
        else:
            risk = "HIGH"

        return Response({
            "prediction": round(prediction, 2),
            "confidence": round(service.confidence, 2),
            "volatility": volatility,
            "risk": risk
        })


class CheckPredictionAPIView(APIView):
    def post(self, request):
        predicted = int(request.data.get("predicted"))
        actual = int(request.data.get("actual"))

        diff = round(abs(actual - predicted), 2)
        status = "PASS" if diff <= PREDICTION_TOLERANCE else "FAIL"

        record = PredictionCheck.objects.create(
            predicted=predicted,
            actual=actual,
            difference=diff,
            status=status
        )

        return Response({
            "status": status,
            "difference": diff,
            "predicted": predicted,
            "actual": actual,
            "time": record.created_at.strftime("%H:%M:%S")
        })



class VolatilityHeatmapView(APIView):
    def my_vol(request):
        if request.Method == 'GET':        
            values = list(
                CrashResult.objects.values_list("multiplier", flat=True)
            )

            window = 10
            heatmap = []

            for i in range(len(values)):
                if i < window - 1:
                    heatmap.append(None)
                else:
                    slice_ = values[i - window + 1:i + 1]
                    heatmap.append(round(statistics.stdev(slice_), 2))

            return Response({"volatility": heatmap})
        else:
            pass


    def export_csv(request):
        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = 'attachment; filename="aviator_data.csv"'
        writer = csv.writer(response)
        writer.writerow(["multiplier", "timestamp"])

        for r in CrashResult.objects.all():
            writer.writerow([r.multiplier, r.timestamp])

        return response


