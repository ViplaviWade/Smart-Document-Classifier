import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# Load the dataset
df = pd.read_csv("document_dataset.csv")

# Convert categorical labels to numeric labels
category_mapping = {label: i for i, label in enumerate(df["category"].unique())}
df["labels"] = df["category"].map(category_mapping)  # Assign numeric values

# Convert Pandas DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df.drop(columns=["category"]))  # Drop original text labels

# Split into training and test sets
dataset = dataset.train_test_split(test_size=0.2)

# Load pre-trained model and tokenizer
model_name = "facebook/bart-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(category_mapping),  # Ensure correct number of labels
    ignore_mismatched_sizes=True  # Ignore weight mismatch errors
)

# Tokenize input data
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

# Apply tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set correct format
tokenized_datasets.set_format("torch")  # Convert dataset to PyTorch format

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # Fix deprecation warning
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_bart")
tokenizer.save_pretrained("./fine_tuned_bart")

print("✅ Fine-tuning complete and model saved!")
