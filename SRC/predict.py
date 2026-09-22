import joblib

# Saved AI model ko load karna
model_path = "../model/spam_classifier.pkl"

model = joblib.load(model_path)

print("===================================")
print("   AI SPAM MESSAGE CLASSIFIER")
print("===================================")
print()

# User se message lena
message = input("Enter your message: ")

# AI se prediction karwana
prediction = model.predict([message])[0]

print()
print("Your message:")
print(message)

print()
print("AI Prediction:")

if prediction == "spam":
    print("SPAM")
    print("Warning: This message may be unwanted or suspicious.")
else:
    print("NOT SPAM")
    print("This message appears to be normal.")