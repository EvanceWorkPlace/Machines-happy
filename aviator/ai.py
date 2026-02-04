"""
Simple AI helper for Aviator analysis.

This module provides a lightweight heuristic suggestion engine and
placeholders for LangChain/OpenAI integration.
"""
from typing import Dict, Any
import pandas as pd
import numpy as np

def summarize_results(results: pd.DataFrame) -> Dict[str, Any]:
    if results.empty:
        return {'count': 0}
    desc = results['multiplier'].describe()
    lows = (results['multiplier'] < 2.0).sum()
    mediums = ((results['multiplier'] >= 2.0) & (results['multiplier'] < 5.0)).sum()
    highs = (results['multiplier'] >= 5.0).sum()
    return {
        'count': int(desc['count']),
        'mean': float(desc['mean']),
        'std': float(desc['std']) if not np.isnan(desc['std']) else 0.0,
        'min': float(desc['min']),
        'max': float(desc['max']),
        'low_count': int(lows),
        'medium_count': int(mediums),
        'high_count': int(highs),
    }

def generate_suggestion(results: pd.DataFrame) -> Dict[str, Any]:
    """Return a recommended cashout multiplier and confidence score.

    This is a simple heuristic: prefer cashouts around mean but avoid
    very low multipliers if there is a recent streak of lows.
    """
    if results.empty:
        return {'recommended_min': 1.5, 'recommended_max': 2.5, 'confidence': 0.1}

    mean = float(results['multiplier'].mean())
    std = float(results['multiplier'].std()) if not np.isnan(results['multiplier'].std()) else 0.0

    # Basic recommended window around mean, clipped
    low = max(1.1, mean - std)
    high = max(low + 0.1, mean + std * 0.5)

    # Confidence based on amount of data and variance
    n = len(results)
    confidence = min(0.95, 0.1 + 0.8 * (1 - (std / (mean + 1e-6))) * min(1.0, n / 200))
    confidence = float(np.clip(confidence, 0.05, 0.95))

    return {
        'recommended_min': round(low, 2),
        'recommended_max': round(high, 2),
        'confidence': round(confidence, 2),
    }
