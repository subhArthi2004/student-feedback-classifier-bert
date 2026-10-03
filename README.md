# 🎓 Student Feedback Classifier using BERT

A Natural Language Processing (NLP) project that uses **BERT (Bidirectional Encoder Representations from Transformers)** to classify student feedback into three categories:

- 🟢 **Positive**
- 🔴 **Negative**
- 🟡 **Neutral**

The project fine-tunes a pre-trained `bert-base-uncased` model using a student feedback dataset and provides prediction functionality for new feedback.

---

## 📌 Project Overview

Student feedback contains valuable information about teaching quality, course content, laboratory sessions, assignments, examinations, faculty, and other academic aspects.

Manually analyzing a large number of feedback responses can be time-consuming. This project uses **BERT-based text classification** to automatically determine the sentiment of student feedback.

### Workflow

```text
Student Feedback
       ↓
Text Tokenization
       ↓
BERT Tokenizer
       ↓
Fine-Tuned BERT Model
       ↓
Classification Layer
       ↓
┌──────────┬──────────┬──────────┐
│ Positive │ Neutral  │ Negative │
└──────────┴──────────┴──────────┘
       ↓
Prediction + Confidence Score
```

---

## 🎯 Objectives

- Develop a BERT-based NLP classification model.
- Classify student feedback into Positive, Negative, and Neutral categories.
- Fine-tune a pre-trained BERT model using student feedback data.
- Evaluate the model using standard classification metrics.
- Predict the sentiment of new student feedback.
- Demonstrate the use of Transformer-based models for academic feedback analysis.

---

## 🧠 Model

The project uses:

**Model:** `bert-base-uncased`

BERT is a Transformer-based language model that understands the contextual relationships between words in a sentence.

Instead of training a language model from scratch, this project **fine-tunes a pre-trained BERT model** for the specific task of student feedback classification.

### Classification Classes

| Label | Description |
|---|---|
| Negative | Feedback expressing dissatisfaction or criticism |
| Neutral | Feedback that is balanced, average, or neither clearly positive nor negative |
| Positive | Feedback expressing satisfaction or appreciation |

---

## 📊 Dataset

The project uses a student feedback dataset containing feedback related to different academic aspects, including:

- Professor
- Course
- Laboratory sessions
- Teaching assistants
- Online classes
- Examination pattern
- Curriculum
- Faculty
- Assignments
- Grading system
- Course material
- Workshops

The dataset contains **288 synthetic student-feedback samples** distributed across the three sentiment categories.

> **Note:** The dataset is synthetic and was created for educational/project purposes. Therefore, the evaluation results should not be interpreted as representative of real-world student feedback.

---

## 🛠️ Technologies Used

- **Python**
- **PyTorch**
- **Hugging Face Transformers**
- **BERT**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Google Colab**
- **CUDA / GPU acceleration**

---

## 📁 Project Structure

```text
student-feedback-classifier-bert/
│
├── BERT_model.ipynb
├── make_dataset.py
├── train_bert.py
├── predict.py
├── student_feedback.csv
├── requirements.txt
└── README.md
```

### File Description

| File | Description |
|---|---|
| `BERT_model.ipynb` | Complete Google Colab workflow for training and evaluating the model |
| `make_dataset.py` | Generates the student feedback dataset |
| `train_bert.py` | Fine-tunes the BERT model |
| `predict.py` | Performs sentiment prediction on new feedback |
| `student_feedback.csv` | Student feedback dataset |
| `requirements.txt` | Required Python libraries |
| `README.md` | Project documentation |

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/subhArthi2004/student-feedback-classifier-bert.git
```

Navigate to the project directory:

```bash
cd student-feedback-classifier-bert
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### 1. Generate the Dataset

Run:

```bash
python make_dataset.py
```

This generates:

```text
student_feedback.csv
```

---

### 2. Train the BERT Model

Run:

```bash
python train_bert.py
```

The model is trained using:

```text
Model              : bert-base-uncased
Maximum Length     : 64
Batch Size         : 16
Epochs             : 4
Learning Rate      : 2e-5
Validation Split   : 20%
```

After training, the fine-tuned model is saved in:

```text
saved_model/
```

---

### 3. Predict New Feedback

Run:

```bash
python predict.py
```

The program accepts student feedback and returns:

```text
Predicted Label
Confidence Score
```

Example:

```text
Input:
The professor explained everything clearly and the course was very engaging.

Prediction:
Positive

Confidence:
75%
```

---

## 📈 Model Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Classification Report
- Confusion Matrix

### Validation Results

| Metric | Score |
|---|---:|
| Accuracy | **100%** |
| Precision | **100%** |
| Recall | **100%** |
| F1-score | **100%** |

The validation set contained **58 samples**:

| Class | Samples |
|---|---:|
| Negative | 19 |
| Neutral | 19 |
| Positive | 20 |
| **Total** | **58** |

### Classification Report

```text
              precision    recall  f1-score   support

Negative          1.00      1.00      1.00        19
Neutral           1.00      1.00      1.00        19
Positive          1.00      1.00      1.00        20

accuracy                              1.00        58
macro avg         1.00      1.00      1.00        58
weighted avg      1.00      1.00      1.00        58
```

> **Important:** The 100% validation performance was obtained on the project's synthetic dataset. It should not be interpreted as 100% accuracy on real-world student feedback.

---

## 🔍 Example Predictions

### 🟢 Positive

```text
"The professor explained everything clearly and the course was very engaging."
```

**Prediction:** Positive

### 🔴 Negative

```text
"The lab sessions were a complete waste of time, badly organized."
```

**Prediction:** Negative

### 🟡 Neutral

```text
"The assignments were okay, nothing special."
```

**Prediction:** Neutral

---

## 🧩 Methodology

The project follows these major steps:

### 1. Dataset Preparation
Student feedback samples are generated and assigned sentiment labels.

### 2. Data Splitting
The dataset is divided into training and validation sets using a stratified split.

### 3. Tokenization
The BERT tokenizer converts textual feedback into tokens and numerical representations.

### 4. BERT Fine-Tuning
The pre-trained `bert-base-uncased` model is fine-tuned for three-class classification.

### 5. Classification
The model predicts one of:

```text
Negative
Neutral
Positive
```

### 6. Evaluation
The trained model is evaluated using accuracy, precision, recall, F1-score, classification report, and confusion matrix.

### 7. Prediction
The trained model can classify new student feedback and provide a confidence score.

---

## ⚠️ Limitations

- The dataset is relatively small.
- The dataset is synthetic rather than collected from real students.
- The model may not generalize well to completely different real-world feedback.
- Complex language, sarcasm, ambiguity, and negation may affect predictions.
- The current classification focuses on overall sentiment rather than individual feedback aspects.

---

## 🔮 Future Scope

The project can be further improved by:

- Using a larger real-world student feedback dataset.
- Supporting multilingual student feedback.
- Implementing aspect-based sentiment analysis.
- Identifying specific academic issues from feedback.
- Adding course-wise and faculty-wise feedback analysis.
- Developing a feedback analytics dashboard.
- Deploying the model as a web or mobile application.
- Continuously improving the model using newly collected feedback.

---

## 🎓 Academic Application

The system can potentially assist educational institutions in analyzing large volumes of student feedback and identifying overall sentiment trends related to:

- Teaching quality
- Course content
- Laboratory sessions
- Assignments
- Examinations
- Faculty
- Course materials
- Workshops

---

## 📚 Key Concepts

This project demonstrates practical implementation of:

- Natural Language Processing
- Text Classification
- Sentiment Analysis
- Transformer Architecture
- BERT
- Transfer Learning
- Fine-Tuning
- Deep Learning
- Model Evaluation

---

## ⭐ Acknowledgment

This project was developed as an academic mini-project to explore **BERT-based Natural Language Processing and sentiment classification**.
