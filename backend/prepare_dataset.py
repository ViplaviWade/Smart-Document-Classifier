import os
import csv
import fitz  # PyMuPDF for PDF text extraction
import docx  # python-docx for Word document text extraction

# Path to the uploaded files and output CSV
UPLOAD_DIR = "uploaded_files"
OUTPUT_FILE = "document_dataset.csv"  # Save in the parent folder

def extract_text(file_path):
    """Extract text from TXT, PDF, or DOCX files."""
    ext = os.path.splitext(file_path)[1].lower()
    print(f"🔍 Processing file: {file_path} (Type: {ext})")  # Debugging log

    try:
        if ext == ".pdf":
            doc = fitz.open(file_path)
            text = "\n".join([page.get_text() for page in doc])
            return text

        elif ext == ".docx":
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text

        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        else:
            print(f"❌ Unsupported file format: {ext}")
            return None

    except Exception as e:
        print(f"❌ Error extracting text from {file_path}: {str(e)}")
        return None

# Categories for classification
categories = {
    "Agreement": "Legal Document",
    "Proposal": "Business Proposal",
    "Paper": "Academic Paper",
    "Automation": "Business Proposal",
}

# Writing header to CSV
with open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["text", "category"])

    # Iterate through uploaded files
    for file_name in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, file_name)

        # Extract text
        text = extract_text(file_path)
        if not text:
            continue  # Skip files with no valid text

        # Assign category based on filename keywords
        assigned_category = None
        for keyword, category in categories.items():
            if keyword.lower() in file_name.lower():
                assigned_category = category
                break

        if not assigned_category:
            assigned_category = "Other"  # Default category

        # Debugging log
        print(f"✅ Extracted Text: {text[:100]}...")  # Show the first 100 characters
        print(f"✅ Assigned Category: {assigned_category}\n")

        # Write to CSV
        writer.writerow([text, assigned_category])
        print(f"✅ Written to CSV: {OUTPUT_FILE}")

print(f"\n✅ Dataset preparation complete. Output saved to {OUTPUT_FILE}")
