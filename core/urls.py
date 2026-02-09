from django.urls import path
from .views import (
    dashboard,
    StatsAPIView,
    CrashResultListCreate,
    PredictAPIView,
    CheckPredictionAPIView,
    VolatilityHeatmapView,
    
)

urlpatterns = [
    path("", dashboard),
    path("api/results/", CrashResultListCreate.as_view()),
    path("api/predict/", PredictAPIView.as_view()),
    path("api/stats/", StatsAPIView.as_view()),
    path("api/check/", CheckPredictionAPIView.as_view()),
    path("api/volatility-heatmap/", VolatilityHeatmapView.as_view()),

]
