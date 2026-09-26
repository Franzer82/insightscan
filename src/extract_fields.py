import re
from pathlib import Path
from PIL import Image
import pytesseract

DATA_DIR = Path("data/SROIE2019/train")
IMG_DIR = DATA_DIR / "img"


def extract_text(image_path: Path) -> str:
    """Führt OCR auf einem Belegbild aus und gibt den Rohtext zurück."""
    image = Image.open(image_path)
    return pytesseract.image_to_string(image, lang="eng")


def extract_total(text: str) -> str | None:
    """Sucht den Gesamtbetrag. Bekannte OCR-Verwechslungen (Komma statt Punkt) werden korrigiert."""
    # Suche nach Zeilen mit "Total" gefolgt von einer Zahl
    matches = re.findall(r"total\D{0,15}?(\d{1,4}[.,]\d{2})", text, re.IGNORECASE)
    if not matches:
        return None
    # Meist ist der LETZTE Treffer der eigentliche Gesamtbetrag (nicht Zwischensummen)
    raw_amount = matches[-1]
    # Bekannte OCR-Verwechslung korrigieren: Komma -> Punkt bei Beträgen
    normalized = raw_amount.replace(",", ".")
    return normalized


def extract_date(text: str) -> str | None:
    """Sucht ein Datum im Format TT/MM/JJJJ."""
    match = re.search(r"\b(\d{2}[/\-]\d{2}[/\-]\d{4})\b", text)
    if match:
        return match.group(1).replace("-", "/")
    return None


def extract_company(text: str) -> str | None:
    """Nimmt an, dass der Firmenname meist in einer der ersten Zeilen steht."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    # Firmenname ist oft die zweite nicht-leere Zeile (erste ist oft der Kundenname/Header)
    for line in lines[:5]:
        if any(keyword in line.upper() for keyword in ["SDN BHD", "ENTERPRISE", "TRADING", "STORE"]):
            return line
    return lines[1] if len(lines) > 1 else None


def extract_fields(image_path: Path) -> dict:
    text = extract_text(image_path)
    return {
        "company": extract_company(text),
        "date": extract_date(text),
        "total": extract_total(text),
        "raw_text": text,
    }


if __name__ == "__main__":
    sample_files = sorted(IMG_DIR.glob("*.jpg"))
    first_sample = sample_files[0]

    result = extract_fields(first_sample)

    print(f"=== Extrahierte Felder für {first_sample.stem} ===")
    print(f"Firma:  {result['company']}")
    print(f"Datum:  {result['date']}")
    print(f"Total:  {result['total']}")