import re

# Mehrere Datumsformate abdecken: Punkt (deutsch), Schrägstrich (SROIE-Standard),
# Bindestrich, sowohl mit 2- als auch 4-stelligem Jahr.
DATE_PATTERNS = [
    r"\b(\d{2}[./-]\d{2}[./-]\d{4})\b",
    r"\b(\d{2}[./-]\d{2}[./-]\d{2})\b",
]

# Deutsche und englische Schlüsselwörter, die typischerweise direkt neben
# dem Gesamtbetrag stehen.
TOTAL_KEYWORDS = [
    "gesamtbetrag", "gesamt", "endbetrag", "zu zahlen", "summe",
    "barzahlung", "total", "amount due",
]

AMOUNT_PATTERN = r"\d{1,4}[.,]\d{2}"


def extract_date_fallback(raw_text: str):
    """Sucht per Regex nach einem Datum im Rohtext, unabhängig davon, ob
    LayoutLM das zugehörige Wort als B-DATE erkannt hat. Deckt mehrere
    gängige Formate ab (Punkt/Schrägstrich/Bindestrich, 2-/4-stelliges Jahr)."""
    for pattern in DATE_PATTERNS:
        match = re.search(pattern, raw_text)
        if match:
            raw_date = match.group(1)
            normalized = raw_date.replace(".", "/").replace("-", "/")
            return normalized
    return None


def extract_total_fallback(raw_text: str):
    """Sucht den Gesamtbetrag per Regex, wenn LayoutLM keinen fand.

    Strategie 1 (bevorzugt): Zeilen mit einem bekannten "Total"-Schlüsselwort
    (deutsch oder englisch) durchsuchen und den letzten Betrag darin nehmen.

    Strategie 2 (Fallback der Fallback-Methode): Falls keine Schlüsselwort-Zeile
    gefunden wird, den GRÖSSTEN im gesamten Text vorkommenden Betrag nehmen -
    das ist eine grobe Heuristik (der Gesamtbetrag ist auf Kassenbons meist,
    aber nicht garantiert, der höchste Einzelwert), aber besser als gar kein
    Ergebnis."""
    lines = raw_text.split("\n")

    for line in lines:
        lower_line = line.lower()
        if any(keyword in lower_line for keyword in TOTAL_KEYWORDS):
            amounts_in_line = re.findall(AMOUNT_PATTERN, line)
            if amounts_in_line:
                return amounts_in_line[-1].replace(",", ".")

    all_amounts = re.findall(AMOUNT_PATTERN, raw_text)
    parsed_amounts = []
    for raw_amount in all_amounts:
        try:
            parsed_amounts.append((float(raw_amount.replace(",", ".")), raw_amount))
        except ValueError:
            continue

    if parsed_amounts:
        parsed_amounts.sort(key=lambda pair: pair[0], reverse=True)
        return parsed_amounts[0][1].replace(",", ".")

    return None