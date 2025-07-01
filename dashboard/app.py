import streamlit as st
import pandas as pd
import pickle

# Load model
with open("models/career_model_xgb.pkl", "rb") as f:
    model = pickle.load(f)

# Load label encoder to decode career label
with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Define the features used for training
FEATURES = [
    'GPA', 'Extracurricular_Activities', 'Internships', 'Projects', 'Leadership_Positions',
    'Field_Specific_Courses', 'Research_Experience', 'Coding_Skills', 'Communication_Skills',
    'Problem_Solving_Skills', 'Teamwork_Skills', 'Analytical_Skills',
    'Presentation_Skills', 'Networking_Skills', 'Industry_Certifications'
]

# Streamlit UI
st.set_page_config(page_title="Academic Career Recommender", layout="centered")
st.title("🎓 Academic Career Recommender System")
st.markdown("Fill out the following details and we'll predict the most suitable career path for you:")

# Input sliders
user_input = {}
for feature in FEATURES:
    user_input[feature] = st.slider(f"{feature.replace('_', ' ')}", 0, 10, 5)

# Predict button
if st.button("🔮 Predict My Career Path"):
    input_df = pd.DataFrame([user_input])  # Convert to DataFrame
    prediction = model.predict(input_df)[0]
    predicted_career = label_encoder.inverse_transform([prediction])[0]
    st.success(f"🚀 Recommended Career: **{predicted_career}**")
