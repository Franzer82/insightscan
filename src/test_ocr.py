from pathlib import Path
from PIL import Image
import pytesseract

DATA_DIR = Path("data/SROIE2019/train")
IMG_DIR = DATA_DIR / "img"

sample_files = sorted(IMG_DIR.glob("*.jpg"))
first_sample = sample_files[0]

image = Image.open(first_sample)

# OCR auf dem rohen Bild ausführen (Englisch, da SROIE-Belege auf Englisch/Malaysisch sind)
extracted_text = pytesseract.image_to_string(image, lang="eng")

print("=== Von Tesseract erkannter Text ===")
print(extracted_text)