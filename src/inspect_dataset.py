import json
from pathlib import Path

DATASET_FILE = Path("data/layoutlm_dataset.jsonl")


def load_samples(limit: int = 3):
    """Lädt die ersten paar Beispiele aus der erzeugten Trainingsdatei."""
    samples = []
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            samples.append(json.loads(line))
    return samples


def print_sample(sample: dict):
    """Zeigt ein Beispiel übersichtlich an: nur die Wörter, die ein relevantes Label haben."""
    print(f"\n{'=' * 50}")
    print(f"Beleg: {sample['id']}")
    print(f"{'=' * 50}")

    print(f"Gesamtzahl Wörter: {len(sample['words'])}")

    # Nur die "interessanten" Wörter zeigen (alles außer "O")
    labeled_words = [
        (word, label)
        for word, label in zip(sample["words"], sample["labels"])
        if label != "O"
    ]

    print(f"\nAls relevant markierte Wörter ({len(labeled_words)}):")
    for word, label in labeled_words:
        print(f"  {label:15s} -> '{word}'")

    if not labeled_words:
        print("  (!) Keine Wörter wurden einem Feld zugeordnet - dieser Beleg wäre nutzlos fürs Training")


if __name__ == "__main__":
    samples = load_samples(limit=5)
    for sample in samples:
        print_sample(sample)