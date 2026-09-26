import json
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizerFast, LayoutLMForTokenClassification

from compliance_rules import evaluate_receipt, parse_receipt_date
from field_fallback import extract_date_fallback, extract_total_fallback
from monitoring import log_prediction
from ocr_extraction import extract_raw_text, extract_words_with_boxes, scale_boxes

MODEL_DIR = Path("models/layoutlm-insightscan")
POLICY_INDEX_FILE = Path("data/policy_index.json")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3"

MAX_LENGTH = 512
TOP_K = 3
DEFAULT_CURRENCY = "RM"


# ------------------------------------------------------------------
# TEIL 1: Wörter+Positionen -> Felder (LayoutLM), mit Regex-Fallback
# ------------------------------------------------------------------

def predict_labels(model, tokenizer, words, boxes, id2label):
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    word_ids = encoding.word_ids()

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
    fields = {"COMPANY": [], "DATE": [], "TOTAL": []}

    for i, word in enumerate(words):
        label = word_predictions.get(i, "O")
        for field in fields:
            if label == f"B-{field}" or label == f"I-{field}":
                fields[field].append(word)

    return {field: " ".join(words_list) for field, words_list in fields.items()}


def parse_box_line(line: str):
    parts = line.strip().split(",", 8)
    if len(parts) < 9:
        return None
    coords = list(map(int, parts[:8]))
    text = parts[8]
    x_coords = coords[0::2]
    y_coords = coords[1::2]
    bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
    return bbox, text


def load_words_from_sroie_box_file(box_file: Path, img_width: int, img_height: int):
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


def load_words_from_ocr(image: Image.Image, img_width: int, img_height: int):
    raw_words, raw_boxes = extract_words_with_boxes(image)
    scaled_boxes = scale_boxes(raw_boxes, img_width, img_height)
    return raw_words, scaled_boxes


def extract_receipt_fields_with_fallback(words, boxes, model, tokenizer, id2label, raw_text: str) -> dict:
    """Kombiniert LayoutLM-Erkennung mit einem Regex-Fallback für DATE und TOTAL.

    Der Fallback für DATE greift nicht nur bei leerem Feld, sondern auch, wenn
    das gelieferte Datum sich nicht sinnvoll parsen lässt (z.B. wenn LayoutLM
    fälschlich eine Steuernummer als Datum einstuft)."""
    word_predictions = predict_labels(model, tokenizer, words, boxes, id2label)
    predicted_fields = extract_fields_from_predictions(words, word_predictions)

    company = predicted_fields["COMPANY"]
    date = predicted_fields["DATE"]
    total = predicted_fields["TOTAL"]

    used_fallback = {"company": False, "date": False, "total": False}

    date_is_valid = date and parse_receipt_date(date) is not None
    if not date_is_valid:
        fallback_date = extract_date_fallback(raw_text)
        if fallback_date and parse_receipt_date(fallback_date) is not None:
            date = fallback_date
            used_fallback["date"] = True

    if not total:
        fallback_total = extract_total_fallback(raw_text)
        if fallback_total:
            total = fallback_total
            used_fallback["total"] = True

    return {
        "company": company,
        "date": date,
        "total": total,
        "used_fallback": used_fallback,
    }


# ------------------------------------------------------------------
# TEIL 2: Felder -> Compliance-Check (Regeln + RAG)
# ------------------------------------------------------------------

def load_policy_index():
    with open(POLICY_INDEX_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def find_relevant_chunks(query: str, policy_index: list, embedding_model, top_k: int = TOP_K):
    query_embedding = embedding_model.encode(query)

    scored_chunks = []
    for entry in policy_index:
        chunk_embedding = np.array(entry["embedding"])
        similarity = cosine_similarity(query_embedding, chunk_embedding)
        scored_chunks.append((similarity, entry["text"]))

    scored_chunks.sort(key=lambda pair: pair[0], reverse=True)
    return scored_chunks[:top_k]


def build_prompt(receipt_fields: dict, relevant_chunks: list, rule_results: dict, currency: str) -> str:
    context_text = "\n\n".join(chunk_text for _, chunk_text in relevant_chunks)

    amount_check = rule_results["amount_check"]
    date_check = rule_results["date_check"]

    prompt = f"""Du bist ein Assistent, der Mitarbeitenden das Ergebnis einer bereits
abgeschlossenen Spesenbeleg-Prüfung freundlich und klar erklärt.

Relevante Auszüge aus der Spesenrichtlinie (nur als Hintergrund/Zitatquelle):
---
{context_text}
---

Beleg-Daten:
- Firma: {receipt_fields.get('company', 'unbekannt')}
- Datum: {receipt_fields.get('date', 'unbekannt')}
- Betrag: {receipt_fields.get('total', 'unbekannt')} {currency}

Das Prüfergebnis steht bereits fest (NICHT selbst nachrechnen, nur erklären):
- Betragsprüfung: {amount_check['explanation']}
- Fristprüfung: {date_check['explanation']}

Fasse dieses Ergebnis in 2-3 kurzen Sätzen für den Mitarbeitenden zusammen. Übernimm die
oben genannten Fakten unverändert, verändere keine Zahlen oder Schlussfolgerungen. Antworte
auf Deutsch."""

    return prompt


def query_ollama(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        },
    )
    response.raise_for_status()
    return response.json()["response"]


def run_compliance_check(receipt_fields: dict, embedding_model, currency: str = DEFAULT_CURRENCY):
    rule_results = evaluate_receipt(receipt_fields, currency=currency)

    policy_index = load_policy_index()
    query = f"Beleg über {receipt_fields.get('total')} {currency} von {receipt_fields.get('company')}"
    relevant_chunks = find_relevant_chunks(query, policy_index, embedding_model)

    prompt = build_prompt(receipt_fields, relevant_chunks, rule_results, currency)
    summary = query_ollama(prompt)

    return {
        "rule_results": rule_results,
        "relevant_chunks": relevant_chunks,
        "summary": summary,
    }


# ------------------------------------------------------------------
# TEIL 3: Gebündeltes Laden der Modelle + komplette Pipeline
# ------------------------------------------------------------------

def load_all_models():
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
    layoutlm_model = LayoutLMForTokenClassification.from_pretrained(MODEL_DIR)
    layoutlm_model.eval()

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return {
        "tokenizer": tokenizer,
        "layoutlm_model": layoutlm_model,
        "id2label": layoutlm_model.config.id2label,
        "embedding_model": embedding_model,
    }


def run_pipeline_on_image(image: Image.Image, models: dict, currency: str = DEFAULT_CURRENCY) -> dict:
    """Die vollständige Pipeline für EIN BELIEBIGES Bild: OCR -> LayoutLM
    (mit Regex-Fallback) -> Monitoring-Log -> Compliance-Check (Regeln + RAG)."""
    img_width, img_height = image.size

    words, boxes = load_words_from_ocr(image, img_width, img_height)
    raw_text = extract_raw_text(image)

    receipt_fields = extract_receipt_fields_with_fallback(
        words, boxes, models["layoutlm_model"], models["tokenizer"], models["id2label"], raw_text
    )

    log_prediction(receipt_fields, currency)

    compliance_result = run_compliance_check(receipt_fields, models["embedding_model"], currency=currency)

    return {
        "fields": receipt_fields,
        "rule_results": compliance_result["rule_results"],
        "relevant_chunks": compliance_result["relevant_chunks"],
        "summary": compliance_result["summary"],
        "currency": currency,
    }


if __name__ == "__main__":
    test_img_dir = Path("data/SROIE2019/test/img")
    sample_files = sorted(test_img_dir.glob("*.jpg"))
    sample_path = sample_files[0]

    print(f"Verarbeite Beleg über den OCR-Pfad: {sample_path.name}\n")

    print("Lade alle Modelle ...")
    models = load_all_models()

    print("Führe komplette Pipeline aus ...\n")
    image = Image.open(sample_path)
    result = run_pipeline_on_image(image, models, currency="EUR")

    print("=== Erkannte Felder ===")
    print(f"Firma : {result['fields']['company']}")
    print(f"Datum : {result['fields']['date']} (Fallback genutzt: {result['fields']['used_fallback']['date']})")
    print(f"Betrag: {result['fields']['total']} {result['currency']} "
          f"(Fallback genutzt: {result['fields']['used_fallback']['total']})")

    print("\n=== Regelprüfung ===")
    print(f"Betrag: {result['rule_results']['amount_check']['explanation']}")
    print(f"Frist:  {result['rule_results']['date_check']['explanation']}")

    print("\n=== Zusammenfassung ===")
    print(result["summary"])