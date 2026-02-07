# utils.py
import statistics

WINDOW = 10

def analyze_sequence(values):
    if len(values) < 3:
        return None, "LOW"

    window = values[-WINDOW:]
    mean = statistics.mean(window)
    std = statistics.stdev(window) if len(window) > 1 else 0

    # volatility classification
    if std < 0.5:
        vol = "LOW"
    elif std < 1.5:
        vol = "MEDIUM"
    else:
        vol = "HIGH"

    # naive forecast (educational only)
    forecast = round(mean * 0.9 + window[-1] * 0.1, 2)

    return forecast, vol


def detect_streaks(values, low=2.0, high=5.0, min_len=3):
    streaks = []
    current = None
    length = 0

    for v in values:
        label = (
            "LOW" if v < low else
            "HIGH" if v >= high else
            "MID"
        )

        if label == current:
            length += 1
        else:
            if current and length >= min_len:
                streaks.append({
                    "type": f"{current}_STREAK",
                    "length": length
                })
            current = label
            length = 1

    if current and length >= min_len:
        streaks.append({
            "type": f"{current}_STREAK",
            "length": length
        })

    return streaks



