import json
from pathlib import Path

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from compliance_rules import evaluate_receipt

POLICY_INDEX_FILE = Path("data/policy_index.json")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3"

TOP_K = 3


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


def build_prompt(receipt_fields: dict, relevant_chunks: list, rule_results: dict) -> str:
    """Baut den Prompt für gemma3. WICHTIG: Das Modell soll NICHT selbst rechnen oder
    entscheiden, ob eine Genehmigung nötig ist - dieses Ergebnis liefern wir bereits
    fertig berechnet mit (siehe rule_results). Das Modell soll nur noch eine
    verständliche, freundliche Erklärung daraus formulieren und dabei auf die
    Richtlinie Bezug nehmen."""
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
- Betrag: {receipt_fields.get('total', 'unbekannt')} RM

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


def check_compliance(receipt_fields: dict):
    print("Führe deterministische Regelprüfung durch ...")
    rule_results = evaluate_receipt(receipt_fields)

    print("\n=== Deterministisches Prüfergebnis (Python-Code, kein LLM) ===")
    print(f"Betrag: {rule_results['amount_check']['explanation']}")
    print(f"Frist:  {rule_results['date_check']['explanation']}")

    print("\nLade Richtlinien-Index und Embedding-Modell ...")
    policy_index = load_policy_index()
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    query = f"Beleg über {receipt_fields.get('total')} RM von {receipt_fields.get('company')}"
    relevant_chunks = find_relevant_chunks(query, policy_index, embedding_model)

    print("\n=== Gefundene relevante Richtlinien-Abschnitte ===")
    for similarity, text in relevant_chunks:
        first_line = text.split("\n")[0]
        print(f"  [Ähnlichkeit: {similarity:.3f}] {first_line}")

    prompt = build_prompt(receipt_fields, relevant_chunks, rule_results)

    print("\nFrage gemma3 nach einer verständlichen Zusammenfassung ...")
    answer = query_ollama(prompt)

    print("\n=== Finale Antwort von gemma3 (basiert auf vorgegebenem Prüfergebnis) ===")
    print(answer)


if __name__ == "__main__":
    example_receipt = {
        "company": "OJC MARKETING SDN BHD",
        "date": "15/01/2019",
        "total": "193.00",
    }
    check_compliance(example_receipt)