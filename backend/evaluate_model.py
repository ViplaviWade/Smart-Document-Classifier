import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Load dataset
dataset = pd.read_csv("document_dataset.csv")

# Print first few rows for debugging
print(f"\n✅ Loaded {len(dataset)} samples")
print(dataset.head())

custom_model_path = "models/bart-large-mnli"

# Load Pre-Trained Model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")

model.save_pretrained(custom_model_path)
tokenizer.save_pretrained(custom_model_path)

# Define categories
CATEGORIES = ["Legal Document", "Business Proposal", "Academic Paper", "General Article", "Other"]

# Lists for actual and predicted labels
actual_labels = []
predicted_labels = []

# Run model on test data
for index, row in dataset.iterrows():
    text = row["text"]
    actual_labels.append(row["category"])

    # Predict category using the ML model
    classification = classifier(text, CATEGORIES, multi_label=False)

    # Get the top predicted category
    predicted_category = classification["labels"][0]
    predicted_labels.append(predicted_category)

# Calculate Accuracy
accuracy = accuracy_score(actual_labels, predicted_labels)
print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")

# Print Classification Report (Fixing the Zero Precision Warning)
print("\n📊 Classification Report:\n", classification_report(actual_labels, predicted_labels, zero_division=1))
