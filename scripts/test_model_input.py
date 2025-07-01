import pickle
import pandas as pd

# Load trained model
with open("../models/career_model_xgb.pkl", "rb") as f:
    model = pickle.load(f)

# Load the Field LabelEncoder
with open("../models/field_label_encoder.pkl", "rb") as f:
    field_encoder = pickle.load(f)

# Choose a field to test (must match what was used in training: e.g., 'Engineering', 'Arts', etc.)
test_field = "Engineering"
field_label = field_encoder.transform([test_field])[0]  # Encode the field

# Build sample input with Field_Label added
sample_input = pd.DataFrame([{
    'GPA': 9, 'Extracurricular_Activities': 8, 'Internships': 9,
    'Projects': 8, 'Leadership_Positions': 7, 'Field_Specific_Courses': 9,
    'Research_Experience': 7, 'Coding_Skills': 9, 'Communication_Skills': 6,
    'Problem_Solving_Skills': 8, 'Teamwork_Skills': 8, 'Analytical_Skills': 9,
    'Presentation_Skills': 7, 'Networking_Skills': 5, 'Industry_Certifications': 6,
    'Field_Label': field_label  # 💥 Add this feature to match model input
}])

# Predict
prediction = model.predict(sample_input)[0]
print("🔮 Predicted Career Label:", prediction)
