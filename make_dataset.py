"""
Generates a synthetic student feedback dataset for the classifier.
Labels: Positive, Negative, Neutral
Run once: python make_dataset.py  -> creates student_feedback.csv
"""
import csv
import random

random.seed(42)

subjects = ["the professor", "this course", "the lab sessions", "the teaching assistant",
            "the online classes", "the exam pattern", "the curriculum", "the faculty",
            "the assignments", "the grading system", "the course material", "the workshop"]

positive_templates = [
    "I really enjoyed {s}, it was very helpful and well organized.",
    "{S} was excellent and improved my understanding a lot.",
    "Great experience with {s}, clear explanations throughout.",
    "{S} exceeded my expectations, very engaging and useful.",
    "I appreciate how well {s} was structured and delivered.",
    "{S} helped me learn a lot, very satisfied with it.",
    "Fantastic job, {s} was informative and easy to follow.",
    "{S} was one of the best parts of this semester.",
]

negative_templates = [
    "I was disappointed with {s}, it needs major improvement.",
    "{S} was confusing and poorly organized.",
    "Not satisfied with {s}, it felt rushed and incomplete.",
    "{S} was a waste of time, very unclear explanations.",
    "I did not like {s} at all, it needs to be redesigned.",
    "{S} was frustrating, lacked proper structure.",
    "Terrible experience with {s}, very unhelpful.",
    "{S} needs a lot of work, it was below expectations.",
]

neutral_templates = [
    "{S} was okay, nothing special but not bad either.",
    "I have mixed feelings about {s}, some parts were fine.",
    "{S} was average, could be better or worse.",
    "{S} was fine overall, no strong opinion either way.",
    "It was an average experience with {s}.",
    "{S} was acceptable, though a few things could improve.",
    "Nothing much to say about {s}, it was moderate.",
    "{S} met basic expectations, nothing more nothing less.",
]

def fill(template, subject):
    return template.format(s=subject, S=subject[0].upper() + subject[1:])

rows = []
for subject in subjects:
    for t in positive_templates:
        rows.append((fill(t, subject), "Positive"))
    for t in negative_templates:
        rows.append((fill(t, subject), "Negative"))
    for t in neutral_templates:
        rows.append((fill(t, subject), "Neutral"))

random.shuffle(rows)

with open("student_feedback.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["feedback_text", "label"])
    writer.writerows(rows)

print(f"Created student_feedback.csv with {len(rows)} rows")
