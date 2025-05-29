import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Load the dataset
with open("data/train_data.json", "r") as f:
    data = json.load(f)

# Extract text and emotion
samples = []
for dialogue in data:
    for utterance in dialogue["dialogue"]:
        text = utterance["transcript"]
        emotion = utterance["emotion"]
        samples.append((text, emotion))

# Create DataFrame
df = pd.DataFrame(samples, columns=["text", "emotion"])

# Vectorize text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["emotion"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Report
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
