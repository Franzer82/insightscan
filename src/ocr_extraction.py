from collections import defaultdict

import pytesseract
from pytesseract import Output
from PIL import Image

TESSERACT_CONFIG = "--psm 6"


def extract_words_with_boxes(image: Image.Image, min_confidence: int = 30):
    """Nutzt Tesseract, um aus einem beliebigen Belegbild Wörter UND deren
    Positionen zu extrahieren. Wörter werden zu ZEILEN gruppiert, und alle
    Wörter einer Zeile bekommen dieselbe (die gesamte Zeile umspannende)
    Box zugewiesen - das entspricht der Struktur unserer Trainingsdaten."""
    data = pytesseract.image_to_data(
        image, lang="eng", config=TESSERACT_CONFIG, output_type=Output.DICT
    )

    lines = defaultdict(list)
    num_entries = len(data["text"])

    for i in range(num_entries):
        text = data["text"][i].strip()
        confidence = int(data["conf"][i])

        if not text or confidence < min_confidence:
            continue

        line_key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])

        left = data["left"][i]
        top = data["top"][i]
        width = data["width"][i]
        height = data["height"][i]
        bbox = [left, top, left + width, top + height]

        lines[line_key].append((text, bbox))

    words = []
    boxes = []

    for line_key in sorted(lines.keys()):
        line_words = lines[line_key]

        all_x1 = [box[0] for _, box in line_words]
        all_y1 = [box[1] for _, box in line_words]
        all_x2 = [box[2] for _, box in line_words]
        all_y2 = [box[3] for _, box in line_words]

        line_bbox = [min(all_x1), min(all_y1), max(all_x2), max(all_y2)]

        for text, _ in line_words:
            words.append(text)
            boxes.append(line_bbox)

    return words, boxes


def extract_raw_text(image: Image.Image) -> str:
    """Liefert den kompletten erkannten Text als einfachen String (Lesereihenfolge,
    mit Zeilenumbrüchen). Wird für den Regex-Fallback genutzt, wenn LayoutLM ein
    Feld nicht erkennen konnte - unabhängig von der Wort-für-Wort-Klassifikation."""
    return pytesseract.image_to_string(image, lang="eng", config=TESSERACT_CONFIG)


def scale_boxes(boxes: list, img_width: int, img_height: int) -> list:
    """Skaliert Pixel-Koordinaten auf die von LayoutLM erwartete 0-1000 Skala."""
    scaled = []
    for bbox in boxes:
        scaled.append([
            int(1000 * bbox[0] / img_width),
            int(1000 * bbox[1] / img_height),
            int(1000 * bbox[2] / img_width),
            int(1000 * bbox[3] / img_height),
        ])
    return scaled