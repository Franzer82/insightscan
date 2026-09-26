import json
from pathlib import Path
from extract_fields import extract_fields

DATA_DIR = Path("data/SROIE2019/train")
IMG_DIR = DATA_DIR / "img"
ENTITIES_DIR = DATA_DIR / "entities"


def load_ground_truth(sample_id: str) -> dict:
    """Lädt die echten (korrekten) Werte aus der entities-Datei."""
    entities_file = ENTITIES_DIR / f"{sample_id}.txt"
    with open(entities_file, "r", encoding="utf-8", errors="ignore") as f:
        return json.load(f)


def normalize_total(value: str | None) -> str | None:
    """Bringt Beträge auf ein einheitliches Format, damit '9.00' und '9' als gleich zählen."""
    if value is None:
        return None
    try:
        return f"{float(value):.2f}"
    except ValueError:
        return value


def compare_fields(predicted: dict, ground_truth: dict) -> dict:
    """Vergleicht Vorhersage gegen Ground Truth, Feld für Feld."""
    results = {}

    predicted_total = normalize_total(predicted.get("total"))
    truth_total = normalize_total(ground_truth.get("total"))
    results["total"] = predicted_total == truth_total

    results["date"] = predicted.get("date") == ground_truth.get("date")

    predicted_company = (predicted.get("company") or "").strip().upper()
    truth_company = (ground_truth.get("company") or "").strip().upper()
    results["company"] = predicted_company == truth_company

    return results


def run_evaluation():
    sample_files = sorted(IMG_DIR.glob("*.jpg"))
    total_samples = len(sample_files)

    correct_counts = {"company": 0, "date": 0, "total": 0}
    failed_samples = []

    for i, image_path in enumerate(sample_files, start=1):
        sample_id = image_path.stem

        try:
            ground_truth = load_ground_truth(sample_id)
        except (FileNotFoundError, json.JSONDecodeError):
            continue

        predicted = extract_fields(image_path)
        comparison = compare_fields(predicted, ground_truth)

        for field, is_correct in comparison.items():
            if is_correct:
                correct_counts[field] += 1
            else:
                failed_samples.append({
                    "sample_id": sample_id,
                    "field": field,
                    "predicted": predicted.get(field),
                    "expected": ground_truth.get(field),
                })

        if i % 50 == 0:
            print(f"Verarbeitet: {i}/{total_samples}")

    print("\n=== Ergebnis der Baseline-Auswertung ===")
    for field, correct in correct_counts.items():
        accuracy = (correct / total_samples) * 100
        print(f"{field:10s}: {correct}/{total_samples} korrekt ({accuracy:.1f}%)")

    return failed_samples


if __name__ == "__main__":
    failures = run_evaluation()

    print("\n=== Beispiele für falsche Extraktionen (erste 10) ===")
    for failure in failures[:10]:
        print(f"[{failure['sample_id']}] Feld '{failure['field']}': "
              f"erkannt='{failure['predicted']}' vs. erwartet='{failure['expected']}'")