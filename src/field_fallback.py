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

    WICHTIG: Belege enthalten oft MEHRERE Zeilen mit einem "Total"-Schlüsselwort
    (z.B. 'Total GST: 0.00', 'Total Inclusive GST: 193.00', 'TOTAL: 193.00') -
    nur eine davon ist der tatsächliche Gesamtbetrag, die anderen sind
    Zwischensummen oder Steueranteile. Ein Fund an der ERSTEN passenden Zeile
    wäre daher unzuverlässig (es könnte zufällig eine Zwischensumme mit
    kleinerem oder sogar 0.00-Betrag sein).

    Strategie: Alle Beträge aus ALLEN Zeilen mit einem Total-Schlüsselwort
    sammeln und den GRÖSSTEN nehmen - der eigentliche Gesamtbetrag ist so gut
    wie immer die größte dieser Zahlen (Zwischensummen und Steueranteile sind
    kleiner oder höchstens gleich groß)."""
    lines = raw_text.split("\n")

    candidate_amounts = []
    for line in lines:
        lower_line = line.lower()
        if any(keyword in lower_line for keyword in TOTAL_KEYWORDS):
            amounts_in_line = re.findall(AMOUNT_PATTERN, line)
            for raw_amount in amounts_in_line:
                try:
                    candidate_amounts.append((float(raw_amount.replace(",", ".")), raw_amount))
                except ValueError:
                    continue

    if candidate_amounts:
        candidate_amounts.sort(key=lambda pair: pair[0], reverse=True)
        return candidate_amounts[0][1].replace(",", ".")

    # Fallback der Fallback-Methode: keine "total"-Zeile gefunden -
    # dann den groessten Betrag im GESAMTEN Text nehmen.
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