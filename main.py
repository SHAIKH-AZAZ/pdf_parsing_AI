import os
from pdf2image import convert_from_path
import pytesseract

# --- FIX FOR ARCH LINUX ---
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
os.environ["TESSDATA_PREFIX"] = "/usr/share/tessdata/"
# --------------------------

# ---------- USER SETTINGS ----------
input_folder = r"sample"
output_folder = r"sample"
# ------------------------------------

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.lower().endswith(".pdf"):
        pdf_path = os.path.join(input_folder, file)
        txt_filename = os.path.splitext(file)[0] + ".txt"
        txt_path = os.path.join(output_folder, txt_filename)

        print(f"Processing: {file}")

        pages = convert_from_path(pdf_path)

        full_text = ""

        for i, page in enumerate(pages):
            print(f" - OCR page {i+1}/{len(pages)}")
            text = pytesseract.image_to_string(page)
            full_text += f"\n--- Page {i+1} ---\n"
            full_text += text

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        print(f"Saved OCR text → {txt_path}\n")

print("✔ OCR conversion completed!")
