import json
from pathlib import Path

import torch
from PIL import Image
from transformers import BertTokenizerFast, LayoutLMForTokenClassification

MODEL_DIR = Path("models/layoutlm-insightscan")
TEST_DIR = Path("data/SROIE2019/test")
TEST_IMG_DIR = TEST_DIR / "img"
TEST_BOX_DIR = TEST_DIR / "box"
TEST_ENTITIES_DIR = TEST_DIR / "entities"

MAX_LENGTH = 512


def parse_box_line(line: str):
    """Identisch zur Aufbereitung beim Training: zerlegt eine box-Zeile
    in ein Rechteck (4 Werte) und den zugehörigen Text."""
    parts = line.strip().split(",", 8)
    if len(parts) < 9:
        return None
    coords = list(map(int, parts[:8]))
    text = parts[8]
    x_coords = coords[0::2]
    y_coords = coords[1::2]
    bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
    return bbox, text


def load_receipt_words(sample_id: str, img_width: int, img_height: int):
    """Liest alle Wörter und deren skalierte Positionen aus der box-Datei
    eines Testbelegs - ohne Labels, die kennen wir hier ja nicht (das ist
    genau das, was das Modell gleich vorhersagen soll)."""
    box_file = TEST_BOX_DIR / f"{sample_id}.txt"

    words = []
    boxes = []

    with open(box_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            parsed = parse_box_line(line)
            if parsed is None:
                continue
            bbox, text = parsed

            scaled_bbox = [
                int(1000 * bbox[0] / img_width),
                int(1000 * bbox[1] / img_height),
                int(1000 * bbox[2] / img_width),
                int(1000 * bbox[3] / img_height),
            ]

            for word in text.split():
                words.append(word)
                boxes.append(scaled_bbox)

    return words, boxes


def predict(model, tokenizer, words, boxes, id2label):
    """Führt die eigentliche Modell-Vorhersage aus und ordnet jedem
    Original-Wort sein vorhergesagtes Label zu (Label-Alignment,
    diesmal in die andere Richtung als beim Training)."""
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    word_ids = encoding.word_ids()

    # bbox muss zur Tensor-Länge passen: Sonderzeichen ([CLS] etc.) bekommen [0,0,0,0]
    aligned_boxes = []
    for word_id in word_ids:
        if word_id is None:
            aligned_boxes.append([0, 0, 0, 0])
        else:
            aligned_boxes.append(boxes[word_id])

    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            bbox=torch.tensor([aligned_boxes]),
        )

    predicted_ids = outputs.logits.argmax(dim=2)[0].tolist()

    # Nur das ERSTE Sub-Token jedes Wortes für die Ausgabe nehmen
    # (Sub-Tokens desselben Wortes wurden beim Training ja auch nur einmal gelabelt)
    word_predictions = {}
    previous_word_id = None
    for token_idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id != previous_word_id:
            word_predictions[word_id] = id2label[predicted_ids[token_idx]]
        previous_word_id = word_id

    return word_predictions


def extract_fields_from_predictions(words, word_predictions):
    """Fasst die Wort-für-Wort-Vorhersagen zu den finalen Feldwerten zusammen
    (mehrere aufeinanderfolgende COMPANY-Wörter werden zu einem String verbunden)."""
    fields = {"COMPANY": [], "DATE": [], "TOTAL": []}

    for i, word in enumerate(words):
        label = word_predictions.get(i, "O")
        for field in fields:
            if label == f"B-{field}" or label == f"I-{field}":
                fields[field].append(word)

    return {field: " ".join(words_list) for field, words_list in fields.items()}


def main():
    print("Lade trainiertes Modell ...")
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
    model = LayoutLMForTokenClassification.from_pretrained(MODEL_DIR)
    model.eval()

    id2label = model.config.id2label

    # Ersten verfügbaren Testbeleg nehmen
    sample_files = sorted(TEST_IMG_DIR.glob("*.jpg"))
    sample_path = sample_files[0]
    sample_id = sample_path.stem

    print(f"\nTeste Beleg: {sample_id} (aus dem Test-Set, NICHT im Training gesehen)\n")

    image = Image.open(sample_path)
    img_width, img_height = image.size

    words, boxes = load_receipt_words(sample_id, img_width, img_height)
    print(f"Anzahl Wörter auf dem Beleg: {len(words)}")

    word_predictions = predict(model, tokenizer, words, boxes, id2label)
    predicted_fields = extract_fields_from_predictions(words, word_predictions)

    print("\n=== Vom Modell vorhergesagt ===")
    for field, value in predicted_fields.items():
        print(f"{field:10s}: {value}")

    # Zum Vergleich: die echte Ground Truth dieses Testbelegs
    entities_file = TEST_ENTITIES_DIR / f"{sample_id}.txt"
    if entities_file.exists():
        with open(entities_file, "r", encoding="utf-8", errors="ignore") as f:
            ground_truth = json.load(f)
        print("\n=== Tatsächliche Ground Truth ===")
        print(f"COMPANY   : {ground_truth.get('company', '(nicht vorhanden)')}")
        print(f"DATE      : {ground_truth.get('date', '(nicht vorhanden)')}")
        print(f"TOTAL     : {ground_truth.get('total', '(nicht vorhanden)')}")


if __name__ == "__main__":
    main()