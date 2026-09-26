import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

POLICY_FILE = Path("data/policies/expense_policy.md")
OUTPUT_FILE = Path("data/policy_index.json")

# Kleines, kostenloses, lokal laufendes Embedding-Modell (ca. 80 MB)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def chunk_policy(text: str) -> list:
    """Zerlegt das Richtliniendokument in einzelne Abschnitte (Chunks).

    Wir nutzen die Markdown-Überschriften ('## ...') als natürliche Trennstellen,
    da jeder Abschnitt der Richtlinie bereits ein inhaltlich abgeschlossenes Thema
    behandelt (Betragsgrenzen, Fristen, etc.) - das ist deutlich sinnvoller als
    z.B. stur alle 500 Zeichen zu trennen, weil dabei zusammengehörige Sätze
    auseinandergerissen werden könnten.
    """
    chunks = []
    current_heading = ""
    current_lines = []

    for line in text.split("\n"):
        if line.startswith("## "):
            # Neuer Abschnitt beginnt - vorherigen Chunk abschließen
            if current_lines:
                chunk_text = current_heading + "\n" + "\n".join(current_lines)
                chunks.append(chunk_text.strip())
            current_heading = line
            current_lines = []
        else:
            current_lines.append(line)

    # Letzten Abschnitt nicht vergessen
    if current_lines:
        chunk_text = current_heading + "\n" + "\n".join(current_lines)
        chunks.append(chunk_text.strip())

    # Leere Chunks (z.B. durch die Hauptüberschrift '# Spesenrichtlinie') herausfiltern
    return [c for c in chunks if len(c.strip()) > 20]


def build_index():
    print(f"Lade Richtliniendokument: {POLICY_FILE}")
    with open(POLICY_FILE, "r", encoding="utf-8") as f:
        policy_text = f.read()

    chunks = chunk_policy(policy_text)
    print(f"Dokument in {len(chunks)} Abschnitte zerlegt:")
    for i, chunk in enumerate(chunks):
        first_line = chunk.split("\n")[0]
        print(f"  [{i}] {first_line}")

    print(f"\nLade Embedding-Modell ({EMBEDDING_MODEL_NAME}) ...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Berechne Embeddings für alle Abschnitte ...")
    embeddings = model.encode(chunks, show_progress_bar=True)

    # Als einfache JSON-Datei speichern: Liste von {text, embedding}
    index_data = [
        {"text": chunk, "embedding": embedding.tolist()}
        for chunk, embedding in zip(chunks, embeddings)
    ]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)

    print(f"\nIndex gespeichert unter: {OUTPUT_FILE}")


if __name__ == "__main__":
    build_index()