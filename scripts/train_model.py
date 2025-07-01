import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load features dataset
df = pd.read_csv("../data/features.csv")

# Define the features to use (drop any text/object columns)
FEATURES = [
    'GPA', 'Extracurricular_Activities', 'Internships', 'Projects', 'Leadership_Positions',
    'Field_Specific_Courses', 'Research_Experience', 'Coding_Skills', 'Communication_Skills',
    'Problem_Solving_Skills', 'Teamwork_Skills', 'Analytical_Skills',
    'Presentation_Skills', 'Networking_Skills', 'Industry_Certifications'
]

X = df[FEATURES]
y = df["Career_Label"]  # Target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("🚀 Training XGBoost Classifier...")
model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
model.fit(X_train, y_train)

# Save model
with open("../models/career_model_xgb.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully.")
