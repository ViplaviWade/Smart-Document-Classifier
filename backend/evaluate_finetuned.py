import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report

# Load the fine-tuned model and tokenizer
model_path = "./fine_tuned_bart"  # Path where the fine-tuned model is saved
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Load test dataset (20% split used during fine-tuning)
df = pd.read_csv("document_dataset.csv")  # Load full dataset

# Convert categorical labels to numerical values (same mapping as before)
category_mapping = {label: i for i, label in enumerate(df["category"].unique())}
df["labels"] = df["category"].map(category_mapping)  # Assign numeric values

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df.drop(columns=["category"]))  # Drop text labels
dataset = dataset.train_test_split(test_size=0.2)  # Use the same 20% test split

# Extract test data
test_texts = dataset["test"]["text"]
actual_labels = dataset["test"]["labels"]

# Tokenize test texts
inputs = tokenizer(test_texts, padding="max_length", truncation=True, return_tensors="pt")

# Ensure model is in evaluation mode
model.eval()

# Disable gradient calculation for efficiency
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits  # Get raw model outputs
    predictions = torch.argmax(logits, dim=-1).numpy()  # Get highest probability class

# Convert predictions back to category names
predicted_labels = [list(category_mapping.keys())[list(category_mapping.values()).index(p)] for p in predictions]
actual_labels_names = [list(category_mapping.keys())[list(category_mapping.values()).index(a)] for a in actual_labels]

# Compute accuracy
accuracy = accuracy_score(actual_labels_names, predicted_labels)
print(f"✅ Fine-Tuned Model Accuracy: {accuracy * 100:.2f}%\n")

# Print detailed classification report
print("📊 Classification Report:")
print(classification_report(actual_labels_names, predicted_labels))
