from pathlib import Path
from PIL import Image

# Pfade zum Trainingsordner
DATA_DIR = Path("data/SROIE2019/train")
IMG_DIR = DATA_DIR / "img"
BOX_DIR = DATA_DIR / "box"
ENTITIES_DIR = DATA_DIR / "entities"

# Ersten Beleg als Beispiel nehmen
sample_files = sorted(IMG_DIR.glob("*.jpg"))
first_sample = sample_files[0]
sample_id = first_sample.stem  # Dateiname ohne Endung

print(f"Beispiel-Beleg: {sample_id}")

# Bild laden und Größe anzeigen
image = Image.open(first_sample)
print(f"Bildgröße: {image.size}")

# Zugehörige Box-Datei (Textkoordinaten) anzeigen
box_file = BOX_DIR / f"{sample_id}.txt"
with open(box_file, "r", encoding="utf-8", errors="ignore") as f:
    box_lines = f.readlines()
print(f"\nAnzahl erkannter Textzeilen im Beleg: {len(box_lines)}")
print("Erste 5 Zeilen der Box-Datei:")
for line in box_lines[:5]:
    print(line.strip())

# Zugehörige Entities-Datei (extrahierte Felder) anzeigen
entities_file = ENTITIES_DIR / f"{sample_id}.txt"
with open(entities_file, "r", encoding="utf-8", errors="ignore") as f:
    print(f"\nExtrahierte Felder (Ground Truth):")
    print(f.read())

# Bild anzeigen (öffnet sich in der Standard-Bildanzeige)
image.show()
