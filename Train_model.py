import pandas as pd
 
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==========================================
# 1. DATASET LOAD KARNA
# ==========================================

file_path = "../data/SMSSpamCollection"

df = pd.read_csv(
    file_path,
    sep="\t",
    header=None,
    names=["label", "message"]
)

print("Dataset loaded successfully!")
print("Total messages:", len(df))
print()


# ==========================================
# 2. INPUT AUR OUTPUT ALAG KARNA
# ==========================================

X = df["message"]
y = df["label"]


# ==========================================
# 3. TRAINING AUR TESTING DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training messages:", len(X_train))
print("Testing messages:", len(X_test))
print()


# ==========================================
# 4. AI PIPELINE BANANA
# ==========================================

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])


# ==========================================
# 5. AI MODEL KO TRAIN KARNA
# ==========================================

print("Training AI model...")
model.fit(X_train, y_train)

print("Model training completed!")
print()


# ==========================================
# 6. TESTING
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 7. MODEL KI ACCURACY
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("================================")
print("MODEL RESULTS")
print("================================")

print("Accuracy:", round(accuracy * 100, 2), "%")
print()


# ==========================================
# 8. DETAILED REPORT
# ==========================================

print("Classification Report:")
print(classification_report(y_test, y_pred))


# ==========================================
# 9. CONFUSION MATRIX
# ==========================================

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ==========================================
# 10. SAMPLE PREDICTIONS
# ==========================================

print()
print("================================")
print("SAMPLE PREDICTIONS")
print("================================")

sample_messages = [
    "Congratulations! You have won a free prize. Click now!",
    "Hey, are you coming to the meeting tomorrow?",
    "URGENT! You have won 50000 rupees. Claim now!",
    "Please call me when you reach home."
]

predictions = model.predict(sample_messages)

for message, prediction in zip(sample_messages, predictions):
    print()
    print("Message:", message)
    print("Prediction:", prediction)
# ==========================================
# 11. TRAINED MODEL SAVE KARNA
# ==========================================

model_path = "../model/spam_classifier.pkl"

joblib.dump(model, model_path)

print()
print("Trained model saved successfully!")
print("Saved at:", model_path)