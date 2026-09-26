import json
from datetime import datetime
from pathlib import Path

LOG_FILE = Path("data/monitoring/prediction_log.jsonl")


def log_prediction(fields: dict, currency: str):
    """Schreibt einen Protokolleintrag für einen verarbeiteten Beleg.

    Wir speichern bewusst NICHT die erkannten Inhalte selbst (Firma, Datum,
    Betrag) - das wären personenbezogene/geschäftliche Daten, die im
    Monitoring nichts verloren haben. Stattdessen speichern wir nur, OB
    LayoutLM erfolgreich war oder der Fallback einspringen musste - das
    reicht für die Qualitäts-Überwachung völlig aus und ist deutlich
    datensparsamer.

    Format: JSONL (JSON Lines) - eine JSON-Zeile pro Eintrag. Das erlaubt
    einfaches Anhängen (kein Neuschreiben der ganzen Datei nötig) und
    zeilenweises Einlesen, auch bei sehr vielen Einträgen.
    """
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "currency": currency,
        "company_recognized": bool(fields.get("company")),
        "date_used_fallback": fields["used_fallback"]["date"],
        "total_used_fallback": fields["used_fallback"]["total"],
        "date_recognized": bool(fields.get("date")),
        "total_recognized": bool(fields.get("total")),
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_logs() -> list:
    """Liest alle bisherigen Protokolleinträge ein. Gibt eine leere Liste
    zurück, falls noch keine Datei existiert (z.B. beim allerersten Start)."""
    if not LOG_FILE.exists():
        return []

    entries = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def compute_stats(logs: list) -> dict:
    """Berechnet zusammenfassende Kennzahlen aus den Protokolleinträgen -
    das Herzstück des Monitorings. Die Fallback-Quote ist der wichtigste
    Wert: Steigt sie deutlich an, ist das ein Hinweis auf 'Data Drift' -
    die eingehenden Belege weichen zunehmend von den Trainingsdaten ab,
    und ein Nachtraining des Modells sollte in Betracht gezogen werden."""
    total = len(logs)

    if total == 0:
        return {
            "total": 0,
            "company_recognition_rate": None,
            "date_fallback_rate": None,
            "total_fallback_rate": None,
        }

    company_recognized_count = sum(1 for entry in logs if entry["company_recognized"])
    date_fallback_count = sum(1 for entry in logs if entry["date_used_fallback"])
    total_fallback_count = sum(1 for entry in logs if entry["total_used_fallback"])

    return {
        "total": total,
        "company_recognition_rate": company_recognized_count / total,
        "date_fallback_rate": date_fallback_count / total,
        "total_fallback_rate": total_fallback_count / total,
    }


def get_recent_fallback_trend(logs: list, window_size: int = 10) -> list:
    """Berechnet die TOTAL-Fallback-Quote über gleitende Fenster von
    'window_size' Belegen - das erlaubt es, einen Trend über Zeit zu
    erkennen (z.B. in einem Liniendiagramm), statt nur eine einzige
    Gesamtquote zu haben, die neue Entwicklungen verschleiern würde."""
    if len(logs) < window_size:
        return []

    trend = []
    for i in range(window_size, len(logs) + 1):
        window = logs[i - window_size:i]
        fallback_count = sum(1 for entry in window if entry["total_used_fallback"])
        trend.append(fallback_count / window_size)

    return trend