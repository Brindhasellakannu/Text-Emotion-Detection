

import joblib

# 1. Load the saved model
print("Loading emotion detection model...")
model = joblib.load("emotion_model.pkl")

# 2. Define emotion names
emotion_labels = {
    0: "Sadness 😢",
    1: "Joy 😊",
    2: "Love ❤️",
    3: "Anger 😠",
    4: "Fear 😨",
    5: "Surprise 😲"
}

print("Model loaded successfully!")
print("\nText Emotion Detection")
print("Type 'exit' to stop.\n")

# 3. Predict emotions from user input
while True:
    text = input("Enter a sentence: ").strip()

    if text.lower() == "exit":
        print("Thank you for testing our model!")
        break

    if not text:
        print("Please enter a sentence.\n")
        continue

    # 4. Predict the emotion
    prediction = model.predict([text])[0]
    emotion = emotion_labels[int(prediction)]

    print("Predicted emotion:", emotion)

    # 5. Display SVM decision scores
    if hasattr(model, "decision_function"):
        scores = model.decision_function([text])[0]

        print("\nEmotion scores:")

        for label, score in sorted(
            zip(emotion_labels.values(), scores),
            key=lambda item: item[1],
            reverse=True
        ):
            print(f"{label}: {score:.3f}")

    print("-" * 35)

