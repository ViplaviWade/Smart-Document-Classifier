import os
import fitz  # PyMuPDF for PDF text extraction
import docx
from fastapi import FastAPI, Depends, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from sqlalchemy.orm import Session
from .database import get_db, engine, Base, SessionLocal
from .models import Document

app = FastAPI()

# Initialize the database tables
Base.metadata.create_all(bind=engine)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load the pre-trained model for classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
print("Model loaded successfully!")

CATEGORIES = [
    "Technical Documentation",
    "Business Proposal",
    "Legal Document",
    "Academic Paper",
    "General Article",
    "Other",
]

def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        try:
            doc = fitz.open(file_path)
            return "\n".join([page.get_text() for page in doc])
        except Exception as e:
            return f"Error reading PDF: {str(e)}"
    elif ext == ".docx":
        try:
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            return f"Error reading DOCX: {str(e)}"
    elif ext == ".txt":
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading TXT: {str(e)}"
    return "Unsupported file format."

@app.post("/upload/")
async def upload_and_classify_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    text = extract_text(file_path)
    if not text:
        raise HTTPException(status_code=400, detail="Failed to extract text from the file")

    # Classify the text
    classification = classifier(text, CATEGORIES, multi_label=False)
    top_category = classification["labels"][0]
    confidence = classification["scores"][0]

    # Store file and classification details in the database
    document = Document(
        filename=file.filename,
        file_path=file_path,
        category=top_category,
        confidence=confidence,
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    return {
        "message": "File uploaded and classified successfully",
        "file_id": document.id,
        "file_path": document.file_path,
        "category": document.category,
        "confidence": document.confidence,
    }
