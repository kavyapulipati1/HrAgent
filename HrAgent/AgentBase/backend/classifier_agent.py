# backend/classifier_agent.py
import os
from transformers import pipeline
from typing import List, Dict

# Disable TensorFlow / Flax imports (important for Windows setups)
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# Default HR categories
DEFAULT_CATEGORIES = [
    "Workplace Conduct",
    "Harassment or Discrimination",
    "Manager Conflict",
    "Scheduling or Shifts",
    "Payroll or Compensation",
    "Leave and PTO",
    "Health and Safety",
    "Performance Management",
    "Facilities or Equipment",
    "Policy Clarification",
    "Retaliation or Whistleblowing"
]

class GrievanceClassifier:
    """Multi-label classifier for HR grievances using zero-shot learning."""

    def __init__(self, categories: List[str] = None, threshold: float = 0.35):
        self.categories = categories or DEFAULT_CATEGORIES
        self.threshold = threshold
        print("🚀 Loading zero-shot classifier model (facebook/bart-large-mnli)...")
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        print("✅ Classifier loaded successfully.")

    def classify(self, grievance: str, categories: List[str] = None) -> List[Dict[str, float]]:
        """Classify a grievance text into one or more HR categories."""
        labels = categories or self.categories
        result = self.classifier(sequences=grievance, candidate_labels=labels, multi_label=True)

        predictions = [
            {"label": label, "score": float(score)}
            for label, score in zip(result["labels"], result["scores"])
            if score >= self.threshold
        ]
        predictions.sort(key=lambda x: x["score"], reverse=True)
        return predictions


# Run a test directly
if __name__ == "__main__":
    grievance = "I haven’t received my salary for two months."
        
    clf = GrievanceClassifier()
    result = clf.classify(grievance)
    print("\n📊 Classification Results:")
    for r in result:
        print(f"{r['label']}: {r['score']:.2f}")
