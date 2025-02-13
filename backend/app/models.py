from sqlalchemy import Column, Integer, String, Float, TIMESTAMP
from .database import Base
from datetime import datetime

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    category = Column(String(255), nullable=True)  # Will store classification result
    confidence = Column(Float, nullable=True)  # Confidence score
    uploaded_at = Column(TIMESTAMP, default=datetime.utcnow)  # Upload timestamp
    file_path = Column(String(255), nullable=False)  # Path to store the file
