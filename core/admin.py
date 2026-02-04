from django.contrib import admin
from .models import CrashResult


@admin.register(CrashResult)
class CrashResultAdmin(admin.ModelAdmin):
    list_display = ('round_id', 'multiplier', 'timestamp')
    list_filter = ('timestamp',)
