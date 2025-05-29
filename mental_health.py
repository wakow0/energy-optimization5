import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
import random
import string

# 1. Simulate some text data and audio features
np.random.seed(42)

def generate_dummy_text():
    words = ["happy", "sad", "anxious", "joyful", "depressed", "angry", "calm", "fearful", "content", "nervous"]
    return ' '.join(random.choices(words, k=20))

# Generate dataset
num_samples = 100
texts = [generate_dummy_text() for _ in range(num_samples)]
audio_features = np.random.rand(num_samples, 13)  # Simulated MFCCs
labels = np.random.randint(0, 2, size=num_samples)  # 0 = no risk, 1 = at-risk

# 2. Text preprocessing using TF-IDF
vectorizer = TfidfVectorizer()
text_features = vectorizer.fit_transform(texts).toarray()

# 3. Combine text and audio features
combined_features = np.hstack((text_features, audio_features))

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(combined_features, labels, test_size=0.2, random_state=42)

# 5. Train model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# 6. Evaluation
report = classification_report(y_test, y_pred, output_dict=True)
acc = accuracy_score(y_test, y_pred)

# 7. Simulated subgroup analysis (gender labels)
gender_labels = ['male' if i % 2 == 0 else 'female' for i in range(num_samples)]
gender_test = [gender_labels[i] for i in X_test[:, 0].argsort()[:len(y_test)]]

# Evaluate by subgroup
subgroup_perf = {}
for gender in set(gender_test):
    idx = [i for i, g in enumerate(gender_test) if g == gender]
    subgroup_perf[gender] = accuracy_score(y_test[idx], y_pred[idx])

import pandas as pd
#import ace_tools as tools; tools.display_dataframe_to_user(name="Classification Report", dataframe=pd.DataFrame(report).transpose())


# Print overall classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print subgroup performance (simulated for gender)
print("\nSubgroup Accuracy (Simulated Gender Labels):")
for gender, score in subgroup_perf.items():
    print(f"{gender}: {score:.2f}")
