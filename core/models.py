from django.db import models


class CrashResult(models.Model):
    """Store a single Aviator crash result as multiplier (e.g., 1.23, 3.5)."""
    round_id = models.CharField(max_length=64, blank=True, null=True)
    multiplier = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.multiplier} @ {self.timestamp.isoformat()}"
