from pathlib import Path
from PIL import Image, ImageOps
import pytesseract

DATA_DIR = Path("data/SROIE2019/train")
IMG_DIR = DATA_DIR / "img"

sample_files = sorted(IMG_DIR.glob("*.jpg"))
first_sample = sample_files[0]

image = Image.open(first_sample)

# Vorverarbeitung: in Graustufen umwandeln, dann Kontrast/Schwellenwert anwenden
gray_image = image.convert("L")
# Autokontrast verstärkt helle/dunkle Bereiche
enhanced_image = ImageOps.autocontrast(gray_image)

# OCR auf dem vorverarbeiteten Bild
extracted_text = pytesseract.image_to_string(enhanced_image, lang="eng")

print("=== Von Tesseract erkannter Text (nach Vorverarbeitung) ===")
print(extracted_text)

# Zum Vergleich: das vorverarbeitete Bild speichern, um es dir anzuschauen
output_path = Path("data/sample_preprocessed.png")
enhanced_image.save(output_path)
print(f"\nVorverarbeitetes Bild gespeichert unter: {output_path}")