"""
Student Feedback Classifier - BERT fine-tuning script.

Trains a bert-base-uncased model to classify student feedback into
Positive / Negative / Neutral using HuggingFace Transformers + PyTorch.

Usage:
    python train_bert.py

Outputs:
    ./saved_model/   -> fine-tuned model + tokenizer (used by predict.py)
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------
CSV_PATH = "student_feedback.csv"
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5
OUTPUT_DIR = "./saved_model"

LABELS = ["Negative", "Neutral", "Positive"]          # fixed order
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for i, label in enumerate(LABELS)}

# ---------------------------------------------------------------------------
# 2. Load & split data
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
df["label_id"] = df["label"].map(label2id)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["feedback_text"].tolist(),
    df["label_id"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label_id"],
)

print(f"Train size: {len(train_texts)}  |  Val size: {len(val_texts)}")

# ---------------------------------------------------------------------------
# 3. Tokenizer & Dataset class
# ---------------------------------------------------------------------------
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)


class FeedbackDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


train_dataset = FeedbackDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = FeedbackDataset(val_texts, val_labels, tokenizer, MAX_LEN)

# ---------------------------------------------------------------------------
# 4. Model
# ---------------------------------------------------------------------------
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id,
)

# ---------------------------------------------------------------------------
# 5. Metrics
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# ---------------------------------------------------------------------------
# 6. Training
# ---------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":
    trainer.train()

    print("\nFinal evaluation:")
    metrics = trainer.evaluate()
    print(metrics)

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}")
