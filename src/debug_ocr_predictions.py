from pathlib import Path

from PIL import Image

from pipeline import load_all_models, load_words_from_ocr, predict_labels

TEST_IMG_DIR = Path("data/SROIE2019/test/img")


def main():
    print("Lade Modelle ...")
    models = load_all_models()

    sample_files = sorted(TEST_IMG_DIR.glob("*.jpg"))
    sample_path = sample_files[0]

    image = Image.open(sample_path)
    img_width, img_height = image.size

    words, boxes = load_words_from_ocr(image, img_width, img_height)
    print(f"\n{len(words)} Wörter von Tesseract erkannt.\n")

    word_predictions = predict_labels(
        models["layoutlm_model"], models["tokenizer"], words, boxes, models["id2label"]
    )

    print("=== Alle Wörter mit vorhergesagtem Label ===")
    for i, word in enumerate(words):
        label = word_predictions.get(i, "?")
        marker = " <---" if label != "O" else ""
        print(f"  [{i:3d}] {label:15s} {word}{marker}")


if __name__ == "__main__":
    main()