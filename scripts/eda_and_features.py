# eda_and_features.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load raw dataset
df = pd.read_csv("../data/raw_dataset.csv")

# Encode the Career column
label_encoder = LabelEncoder()
df["Career_Label"] = label_encoder.fit_transform(df["Career"])

# Save the label encoder for decoding later
import pickle
with open("../models/career_label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Drop text-based categorical columns: Field and Career
df.drop(columns=["Career", "Field"], inplace=True)

# Save preprocessed data for model training
df.to_csv("../data/features.csv", index=False)
print("✅ Features saved to features.csv")
