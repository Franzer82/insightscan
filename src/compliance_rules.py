from datetime import datetime


def parse_receipt_date(date_str: str):
    """Versucht, das Belegdatum in ein echtes Datum umzuwandeln.
    Unsere Belege nutzen unterschiedliche Formate (TT/MM/JJJJ oder TT-MM-JJ),
    deshalb probieren wir mehrere Formate der Reihe nach durch."""
    if not date_str:
        return None

    normalized = date_str.replace("-", "/")
    formats_to_try = ["%d/%m/%Y", "%d/%m/%y"]

    for fmt in formats_to_try:
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue

    return None


def evaluate_amount_rule(total_str: str, currency: str = "RM") -> dict:
    """Prüft den Betrag gegen die Betragsgrenzen aus Abschnitt 2 der Richtlinie.

    'currency' ist rein kosmetisch (wird nur in den Erklärungstexten verwendet) -
    die eigentliche Vergleichslogik arbeitet mit reinen Zahlen und ist damit
    automatisch für jede Währung gültig, ohne dass der Code angepasst werden
    muss."""
    try:
        amount = float(str(total_str).replace(",", "."))
    except (ValueError, TypeError):
        return {
            "amount_known": False,
            "requires_approval": None,
            "approval_level": None,
            "explanation": "Betrag konnte nicht ausgelesen werden - manuelle Prüfung nötig.",
        }

    if amount <= 100.00:
        return {
            "amount_known": True,
            "amount": amount,
            "requires_approval": False,
            "approval_level": "keine",
            "explanation": f"{amount:.2f} {currency} liegt bei oder unter 100,00 {currency} - "
                            f"keine gesonderte Genehmigung nötig.",
        }
    elif amount <= 500.00:
        return {
            "amount_known": True,
            "amount": amount,
            "requires_approval": True,
            "approval_level": "direkte Führungskraft",
            "explanation": f"{amount:.2f} {currency} liegt zwischen 100,01 {currency} und 500,00 {currency} - "
                            f"Freigabe der direkten Führungskraft erforderlich.",
        }
    elif amount <= 2000.00:
        return {
            "amount_known": True,
            "amount": amount,
            "requires_approval": True,
            "approval_level": "Abteilungsleitung",
            "explanation": f"{amount:.2f} {currency} liegt über 500,00 {currency} - zusätzlich zur "
                            f"Führungskraft ist auch die Freigabe der Abteilungsleitung erforderlich.",
        }
    else:
        return {
            "amount_known": True,
            "amount": amount,
            "requires_approval": True,
            "approval_level": "vorherige schriftliche Genehmigung",
            "explanation": f"{amount:.2f} {currency} liegt über 2.000,00 {currency} - eine Erstattung "
                            f"ist ohne vorherige schriftliche Genehmigung vor dem Kauf grundsätzlich "
                            f"NICHT möglich.",
        }


def evaluate_date_rule(date_str: str, reference_date: datetime = None) -> dict:
    """Prüft das Belegdatum gegen die 60/90-Tage-Fristen aus Abschnitt 4 der Richtlinie."""
    if reference_date is None:
        reference_date = datetime.now()

    parsed_date = parse_receipt_date(date_str)
    if parsed_date is None:
        return {
            "date_known": False,
            "within_deadline": None,
            "explanation": "Belegdatum konnte nicht eindeutig gelesen werden - manuelle Prüfung nötig.",
        }

    days_since_purchase = (reference_date - parsed_date).days

    if days_since_purchase < 0:
        return {
            "date_known": True,
            "days_since_purchase": days_since_purchase,
            "within_deadline": None,
            "explanation": "Belegdatum liegt in der Zukunft - bitte Datum prüfen.",
        }
    elif days_since_purchase <= 60:
        return {
            "date_known": True,
            "days_since_purchase": days_since_purchase,
            "within_deadline": True,
            "explanation": f"Beleg ist {days_since_purchase} Tage alt - innerhalb der 60-Tage-Frist.",
        }
    elif days_since_purchase <= 90:
        return {
            "date_known": True,
            "days_since_purchase": days_since_purchase,
            "within_deadline": False,
            "explanation": f"Beleg ist {days_since_purchase} Tage alt - die reguläre 60-Tage-Frist "
                            f"ist überschritten, liegt aber noch innerhalb von 90 Tagen.",
        }
    else:
        return {
            "date_known": True,
            "days_since_purchase": days_since_purchase,
            "within_deadline": False,
            "explanation": f"Beleg ist {days_since_purchase} Tage alt - die 90-Tage-Grenze ist "
                            f"überschritten, eine Erstattung ist grundsätzlich NICHT mehr möglich.",
        }


def evaluate_receipt(receipt_fields: dict, currency: str = "RM") -> dict:
    """Führt alle deterministischen Prüfungen für einen Beleg zusammen.
    'currency' wird durchgereicht und bestimmt nur die Anzeige, nicht die Logik."""
    amount_result = evaluate_amount_rule(receipt_fields.get("total"), currency=currency)
    date_result = evaluate_date_rule(receipt_fields.get("date"))

    return {
        "amount_check": amount_result,
        "date_check": date_result,
    }