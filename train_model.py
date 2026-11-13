import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# 1. Define the emotion labels
emotion_labels = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# 2. Load the datasets
print("Loading datasets...")

train_df = pd.read_parquet(
    "data/train-00000-of-00001.parquet"
)

validation_df = pd.read_parquet(
    "data/validation-00000-of-00001.parquet"
)

# 3. Separate text and labels
X_train = train_df["text"]
y_train = train_df["label"]

X_val = validation_df["text"]
y_val = validation_df["label"]

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))

# 4. Define the models to compare
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),
    "Linear SVM": LinearSVC()
}

results = {}

# 5. Train and evaluate each model
for model_name, classifier in models.items():

    print("\nTraining:", model_name)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=50000
        )),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)

    results[model_name] = accuracy
    if model_name == "Linear SVM":
        joblib.dump(pipeline, "emotion_model.pkl")
        print("Linear SVM model saved successfully!")

    print(f"Validation accuracy: {accuracy:.2%}")

    print("\nClassification report:")
    print(
        classification_report(
            y_val,
            predictions,
            labels=list(emotion_labels.keys()),
            target_names=list(emotion_labels.values()),
            zero_division=0
        )
    )

# 6. Compare the models
print("\n" + "=" * 40)
print("MODEL COMPARISON")
print("=" * 40)

for name, accuracy in sorted(
    results.items(),
    key=lambda item: item[1],
    reverse=True
):
    print(f"{name}: {accuracy:.2%}")

print("\nTraining and validation completed!")
