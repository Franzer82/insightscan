import json
import re
import string
from pathlib import Path
from difflib import SequenceMatcher
from PIL import Image

DATA_DIR = Path("data/SROIE2019/train")
IMG_DIR = DATA_DIR / "img"
BOX_DIR = DATA_DIR / "box"
ENTITIES_DIR = DATA_DIR / "entities"

OUTPUT_FILE = Path("data/layoutlm_dataset.jsonl")

ROW_Y_TOLERANCE = 25


def parse_box_line(line: str):
    """Zerlegt eine Zeile der box-Datei in Koordinaten (Rechteck) und Text."""
    parts = line.strip().split(",", 8)
    if len(parts) < 9:
        return None
    coords = list(map(int, parts[:8]))
    text = parts[8]
    x_coords = coords[0::2]
    y_coords = coords[1::2]
    bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
    return bbox, text


def normalize_text(text: str) -> str:
    """Vereinheitlicht Text für robusteren Vergleich: Kleinschreibung, doppelte Leerzeichen entfernen."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_amount(text: str) -> str:
    """Extrahiert eine reine Geldbetrag-Zahl aus einem Wort, z.B. 'RM9.00' -> '9.00'."""
    match = re.search(r"\d+[.,]\d{2}", text)
    if not match:
        return ""
    return match.group(0).replace(",", ".")


def is_similar(a: str, b: str, threshold: float = 0.85) -> bool:
    """Prüft Textähnlichkeit (0.0 - 1.0). Toleriert kleine Abweichungen statt exaktem Vergleich."""
    return SequenceMatcher(None, a, b).ratio() >= threshold


def label_line_tokens(words: list, field: str) -> list:
    """Vergibt B-/I-Labels an alle Wörter einer kompletten Zeile (für Firma)."""
    labels = []
    for i, _ in enumerate(words):
        labels.append(f"B-{field.upper()}" if i == 0 else f"I-{field.upper()}")
    return labels


def y_center(bbox: list) -> float:
    """Berechnet die vertikale Mitte einer Box - für den Zeilen-Höhenvergleich."""
    return (bbox[1] + bbox[3]) / 2


def find_total_position_based(line_entries: list, normalized_total: str) -> bool:
    """Versuch 1 (bevorzugt): Sucht den Betrag auf ähnlicher Höhe wie eine 'total'-Zeile."""
    total_label_entries = [e for e in line_entries if "total" in normalize_text(e["text"])]

    for total_entry in total_label_entries:
        target_y = y_center(total_entry["bbox"])
        for entry in line_entries:
            if abs(y_center(entry["bbox"]) - target_y) > ROW_Y_TOLERANCE:
                continue
            for i, word in enumerate(entry["words"]):
                if normalize_amount(word) == normalized_total:
                    entry["labels"][i] = "B-TOTAL"
                    return True
    return False


def find_total_fallback(line_entries: list, normalized_total: str) -> bool:
    """Versuch 2 (Fallback): Den gesamten Beleg nach dem exakten Betrag durchsuchen."""
    for entry in line_entries:
        for i, word in enumerate(entry["words"]):
            if normalize_amount(word) == normalized_total:
                entry["labels"][i] = "B-TOTAL"
                return True
    return False


def build_sample(sample_id: str):
    """Baut die vollständigen Trainingsdaten (Wörter, Positionen, Labels) für einen Beleg."""
    box_file = BOX_DIR / f"{sample_id}.txt"
    entities_file = ENTITIES_DIR / f"{sample_id}.txt"
    image_file = IMG_DIR / f"{sample_id}.jpg"

    if not (box_file.exists() and entities_file.exists() and image_file.exists()):
        return None

    with open(entities_file, "r", encoding="utf-8", errors="ignore") as f:
        entities = json.load(f)

    with open(box_file, "r", encoding="utf-8", errors="ignore") as f:
        raw_lines = [parse_box_line(line) for line in f if line.strip()]
    raw_lines = [entry for entry in raw_lines if entry is not None]

    image = Image.open(image_file)
    img_width, img_height = image.size

    line_entries = []
    for bbox, line_text in raw_lines:
        words = line_text.split()
        if not words:
            continue
        line_entries.append({
            "bbox": bbox,
            "text": line_text,
            "words": words,
            "labels": ["O"] * len(words),
        })

    # --- Firma: ganze Zeile vergleichen ---
    for entry in line_entries:
        normalized_line = normalize_text(entry["text"])
        if "company" in entities and is_similar(normalized_line, normalize_text(entities["company"])):
            entry["labels"] = label_line_tokens(entry["words"], "company")

    # --- Datum: einzelnes Wort innerhalb einer noch nicht zugeordneten Zeile ---
    if "date" in entities:
        clean_date = entities["date"].replace("-", "/")
        for entry in line_entries:
            if all(label == "O" for label in entry["labels"]):
                for i, word in enumerate(entry["words"]):
                    clean_word = word.strip(string.punctuation).replace("-", "/")
                    if clean_word == clean_date:
                        entry["labels"][i] = "B-DATE"

    # --- Total: zuerst positionsbasiert versuchen, sonst Fallback über den ganzen Beleg ---
    normalized_total = normalize_amount(entities.get("total", ""))
    if normalized_total:
        success = find_total_position_based(line_entries, normalized_total)
        if not success:
            find_total_fallback(line_entries, normalized_total)

    # --- Finale Liste bauen: Koordinaten skalieren, alles zusammenführen ---
    words, boxes, labels = [], [], []
    for entry in line_entries:
        bbox = entry["bbox"]
        scaled_bbox = [
            int(1000 * bbox[0] / img_width),
            int(1000 * bbox[1] / img_height),
            int(1000 * bbox[2] / img_width),
            int(1000 * bbox[3] / img_height),
        ]
        for word, label in zip(entry["words"], entry["labels"]):
            words.append(word)
            boxes.append(scaled_bbox)
            labels.append(label)

    return {"id": sample_id, "words": words, "bboxes": boxes, "labels": labels}


def build_dataset():
    sample_files = sorted(IMG_DIR.glob("*.jpg"))
    total = len(sample_files)
    written = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for i, image_file in enumerate(sample_files, start=1):
            sample_id = image_file.stem
            sample = build_sample(sample_id)
            if sample is not None and any(label != "O" for label in sample["labels"]):
                out_f.write(json.dumps(sample) + "\n")
                written += 1
            if i % 100 == 0:
                print(f"Verarbeitet: {i}/{total}")

    print(f"\nFertig. {written}/{total} Belege erfolgreich aufbereitet -> {OUTPUT_FILE}")


if __name__ == "__main__":
    build_dataset()