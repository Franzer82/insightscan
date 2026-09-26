import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizerFast,
    LayoutLMConfig,
    LayoutLMForTokenClassification,
    Trainer,
    TrainingArguments,
)
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

DATASET_FILE = Path("data/layoutlm_dataset.jsonl")
BASE_MODEL_DIR = Path("data/SROIE2019/layoutlm-base-uncased")
OUTPUT_MODEL_DIR = Path("models/layoutlm-insightscan")

RANDOM_SEED = 42
VAL_SPLIT = 0.15
MAX_LENGTH = 512


def load_dataset():
    """Liest unsere zuvor erzeugte layoutlm_dataset.jsonl zeilenweise ein."""
    samples = []
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def build_label_list(samples):
    """Sammelt alle im Datensatz vorkommenden Labels (z.B. O, B-COMPANY, I-COMPANY, ...).
    'O' bekommt bewusst Label-ID 0, der Rest wird alphabetisch sortiert - Reihenfolge muss
    beim Training und späteren Laden des Modells immer identisch sein."""
    label_set = set()
    for sample in samples:
        label_set.update(sample["labels"])
    labels = sorted(label_set - {"O"})
    return ["O"] + labels


def split_dataset(samples, val_split=VAL_SPLIT, seed=RANDOM_SEED):
    """Teilt die Belege zufällig (aber reproduzierbar dank fixem seed) in
    Trainings- und Validierungsset auf."""
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    val_size = int(len(shuffled) * val_split)
    return shuffled[val_size:], shuffled[:val_size]


def load_pretrained_layoutlm(config):
    """Lädt das mitgelieferte LayoutLM-Modell manuell.

    Hintergrund: Die Gewichtsdatei stammt aus dem ursprünglichen Microsoft-Forschungscode
    und speichert ihre Parameter intern unter dem Namen 'bert.*' statt des von der
    Hugging-Face-Bibliothek erwarteten 'layoutlm.*'. Dadurch würde die Standard-Lademethode
    (from_pretrained) fälschlich ALLE vortrainierten Gewichte verwerfen und stattdessen ein
    komplett zufällig initialisiertes Modell erzeugen - genau das ist beim ersten
    Trainingsversuch passiert (siehe LOAD REPORT: alles als UNEXPECTED/MISSING markiert).

    Wir laden die Rohdaten daher selbst und benennen die Schlüssel passend um.
    """
    model = LayoutLMForTokenClassification(config)

    weights_path = BASE_MODEL_DIR / "pytorch_model.bin"
    # weights_only=True: Sicherheitsmaßnahme gegen potenziell schädlichen Code beim Laden
    # von Pickle-basierten Dateien (siehe Projekt-Sicherheitsrichtlinie).
    raw_state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

    remapped_state_dict = {}
    for key, value in raw_state_dict.items():
        if key.startswith("bert."):
            new_key = "layoutlm." + key[len("bert."):]
            remapped_state_dict[new_key] = value
        elif key.startswith("cls."):
            # Das ist der ursprüngliche "Masked Language Model"-Kopf des Basismodells -
            # den brauchen wir für unsere Token-Klassifikation nicht, wird übersprungen.
            continue
        else:
            remapped_state_dict[key] = value

    # strict=False: Der neue Klassifikations-Kopf (classifier.weight/bias) existiert im
    # vortrainierten Checkpoint naturgemäß nicht - der bleibt bewusst zufällig initialisiert
    # und wird erst durch unser Fine-Tuning gelernt. Alles andere (Embeddings, 12 Encoder-
    # Schichten) sollte jetzt aber sauber übernommen werden.
    missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)

    print(f"Nicht geladen (erwartet - neuer Klassifikationskopf): {missing_keys}")
    print(f"Ungenutzt aus Checkpoint (erwartet - alter LM-Kopf): {len(unexpected_keys)} Einträge")

    return model


class ReceiptDataset(Dataset):
    """Wandelt unsere Wort/Box/Label-Listen in das Format um, das das Modell erwartet.
    Wichtigster Teil: Label-Alignment (siehe Erklärung im Chat)."""

    def __init__(self, samples, tokenizer, label2id, max_length=MAX_LENGTH):
        self.samples = samples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        words = sample["words"]
        boxes = sample["bboxes"]
        word_labels = sample["labels"]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        word_ids = encoding.word_ids()
        aligned_labels = []
        aligned_boxes = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
                aligned_boxes.append([0, 0, 0, 0])
            elif word_id != previous_word_id:
                aligned_labels.append(self.label2id[word_labels[word_id]])
                aligned_boxes.append(boxes[word_id])
            else:
                aligned_labels.append(-100)
                aligned_boxes.append(boxes[word_id])
            previous_word_id = word_id

        return {
            "input_ids": torch.tensor(encoding["input_ids"]),
            "attention_mask": torch.tensor(encoding["attention_mask"]),
            "bbox": torch.tensor(aligned_boxes),
            "labels": torch.tensor(aligned_labels),
        }


def compute_metrics_builder(id2label):
    """Erzeugt die Funktion, die der Trainer nach jeder Epoche aufruft,
    um Precision/Recall/F1 auf den Validierungsdaten zu berechnen."""

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        true_labels = []
        true_predictions = []

        for pred_row, label_row in zip(predictions, labels):
            row_true = []
            row_pred = []
            for pred_id, label_id in zip(pred_row, label_row):
                if label_id == -100:
                    continue
                row_true.append(id2label[label_id])
                row_pred.append(id2label[pred_id])
            true_labels.append(row_true)
            true_predictions.append(row_pred)

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }

    return compute_metrics


def main():
    print("Lade Datensatz ...")
    samples = load_dataset()
    print(f"{len(samples)} Belege geladen.")

    label_list = build_label_list(samples)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    print(f"Gefundene Labels: {label_list}")

    train_samples, val_samples = split_dataset(samples)
    print(f"Training: {len(train_samples)} Belege, Validierung: {len(val_samples)} Belege")

    print("Lade Tokenizer ...")
    tokenizer = BertTokenizerFast.from_pretrained(BASE_MODEL_DIR)

    print("Baue Modell-Konfiguration und lade vortrainierte Gewichte (mit Namens-Korrektur) ...")
    config = LayoutLMConfig.from_pretrained(
        BASE_MODEL_DIR,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )
    model = load_pretrained_layoutlm(config)

    train_dataset = ReceiptDataset(train_samples, tokenizer, label2id)
    val_dataset = ReceiptDataset(val_samples, tokenizer, label2id)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_MODEL_DIR / "checkpoints"),
        num_train_epochs=15,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=20,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_builder(id2label),
    )

    print("\nStarte Training ...")
    trainer.train()

    print("\nSpeichere finales Modell ...")
    OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_MODEL_DIR))
    tokenizer.save_pretrained(str(OUTPUT_MODEL_DIR))

    print("\n=== Finale Auswertung auf Validierungsdaten ===")
    predictions, labels, _ = trainer.predict(val_dataset)
    predictions = np.argmax(predictions, axis=2)

    true_labels = []
    true_predictions = []
    for pred_row, label_row in zip(predictions, labels):
        row_true = []
        row_pred = []
        for pred_id, label_id in zip(pred_row, label_row):
            if label_id == -100:
                continue
            row_true.append(id2label[label_id])
            row_pred.append(id2label[pred_id])
        true_labels.append(row_true)
        true_predictions.append(row_pred)

    print(classification_report(true_labels, true_predictions))


if __name__ == "__main__":
    main()