# models.py
from django.db import models

class CrashResult(models.Model):
    round_id = models.CharField(max_length=64, blank=True, null=True)
    multiplier = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)

    # analysis fields
    volatility = models.CharField(
        max_length=10,
        choices=[("LOW", "Low"), ("MEDIUM", "Medium"), ("HIGH", "High")],
        blank=True
    )

    class Meta:
        ordering = ["timestamp"]

    def __str__(self):
        return f"{self.multiplier}"

class PredictionCheck(models.Model):
    predicted = models.FloatField()
    actual = models.FloatField()
    difference = models.FloatField()
    status = models.CharField(max_length=10)
    created_at = models.DateTimeField(auto_now_add=True)

