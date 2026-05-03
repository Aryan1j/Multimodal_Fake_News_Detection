import pandas as pd

true_df = pd.read_csv("data/True.csv")
fake_df = pd.read_csv("data/Fake.csv")

true_df["label"] = 1
fake_df["label"] = 0

df = pd.concat([true_df, fake_df], ignore_index=True)

# Add missing title column (IMPORTANT FIX)
df["title"] = ""

# Keep required columns
df = df[["title", "text", "label"]]

df.to_csv("data/test_combined.csv", index=False)

print("Fixed and saved as data/test_combined.csv")