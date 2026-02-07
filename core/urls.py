from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path("api/export/", views.export_csv, name="export_csv"),
    path("api/volatility-heatmap/", views.VolatilityHeatmapView.as_view(), name="volatility_heatmap"),
    path('api/results/', views.CrashResultListCreate.as_view(), name='results_api'),
    path('api/suggestion/', views.SuggestionView.as_view(), name='suggestion_api'),
]
