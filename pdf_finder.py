import os
import shutil

# Source directory where you want to search for PDFs
source_folder = r"resumes"

# Destination directory where all PDFs will be copied
destination_folder = r"final_resume"

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Walk through all folders and subfolders
for root, dirs, files in os.walk(source_folder):
    for file in files:
        if file.lower().endswith(".pdf"):
            source_path = os.path.join(root, file)
            destination_path = os.path.join(destination_folder, file)

            # If duplicate file name exists, create a unique name
            if os.path.exists(destination_path):
                base, ext = os.path.splitext(file)
                count = 1

                while True:
                    new_name = f"{base}_{count}{ext}"
                    new_path = os.path.join(destination_folder, new_name)
                    if not os.path.exists(new_path):
                        destination_path = new_path
                        break
                    count += 1

            # Copy the PDF to destination
            shutil.copy2(source_path, destination_path)
            print(f"Copied: {source_path} → {destination_path}")

print("✔ All PDFs have been copied successfully.")
