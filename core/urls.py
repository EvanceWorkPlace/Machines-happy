from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('api/results/', views.CrashResultListCreate.as_view(), name='results_api'),
    path('api/suggestion/', views.SuggestionView.as_view(), name='suggestion_api'),
]
