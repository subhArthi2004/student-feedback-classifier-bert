"""
Load the fine-tuned BERT model and classify new student feedback.

Usage:
    python predict.py
    (then type feedback text when prompted, or edit SAMPLE_TEXTS below)
"""

import torch
from transformers import BertTokenizerFast, BertForSequenceClassification

MODEL_DIR = "./saved_model"

tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def predict(text: str):
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=64,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).squeeze()
        pred_id = int(torch.argmax(probs))

    label = model.config.id2label[pred_id]
    confidence = float(probs[pred_id])
    return label, confidence


SAMPLE_TEXTS = [
    "The professor explained everything clearly and the course was very engaging.",
    "The lab sessions were a complete waste of time, badly organized.",
    "The assignments were okay, nothing special.",
]

if __name__ == "__main__":
    print("=== Batch predictions on sample texts ===")
    for text in SAMPLE_TEXTS:
        label, conf = predict(text)
        print(f"Text: {text}\nPredicted: {label}  (confidence: {conf:.2f})\n")

    print("=== Type your own feedback (or 'quit' to exit) ===")
    while True:
        user_input = input("\nFeedback: ")
        if user_input.strip().lower() == "quit":
            break
        label, conf = predict(user_input)
        print(f"Predicted: {label}  (confidence: {conf:.2f})")
