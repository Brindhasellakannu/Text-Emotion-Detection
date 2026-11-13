
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Define emotion labels
emotion_labels = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

label_names = [
    emotion_labels[i] for i in range(6)
]

# 2. Load the saved model and test dataset
print("Loading saved model...")
model = joblib.load("emotion_model.pkl")

print("Loading test dataset...")
test_df = pd.read_parquet(
    "data/test-00000-of-00001.parquet"
)

X_test = test_df["text"]
y_test = test_df["label"]

print("Test samples:", len(X_test))

# 3. Predict emotions
print("\nEvaluating model...")
predictions = model.predict(X_test)

# 4. Calculate test accuracy
accuracy = accuracy_score(y_test, predictions)

print(f"\nTest accuracy: {accuracy:.2%}")

# 5. Generate classification report
report = classification_report(
    y_test,
    predictions,
    labels=list(emotion_labels.keys()),
    target_names=label_names,
    zero_division=0
)

print("\nCLASSIFICATION REPORT")
print(report)

# 6. Generate confusion matrix
matrix = confusion_matrix(
    y_test,
    predictions,
    labels=list(emotion_labels.keys())
)

print("CONFUSION MATRIX")
print("Label order:", label_names)
print(matrix)

# 7. Save evaluation results
with open("evaluation_report.txt", "w", encoding="utf-8") as file:
    file.write(f"Test accuracy: {accuracy:.2%}\n\n")
    file.write("CLASSIFICATION REPORT\n")
    file.write(report)
    file.write("\nCONFUSION MATRIX\n")
    file.write(f"Label order: {label_names}\n")
    file.write(str(matrix))

print("\nEvaluation report saved to evaluation_report.txt")
