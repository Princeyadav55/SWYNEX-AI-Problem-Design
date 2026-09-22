import pandas as pd

file_path = "../data/SMSSpamCollection"

df = pd.read_csv(
    file_path,
    sep="\t",
    header=None,
    names=["label", "message"]
)

print("Dataset loaded successfully!")
print()

print("Total messages:", len(df))
print()

print("First 5 messages:")
print(df.head())
print()

print("Label distribution:")
print(df["label"].value_counts())