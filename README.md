# 🎓 Academic Career Recommender System

This project is a complete **Data Science and Machine Learning solution** that recommends the most suitable **academic career path** for a student based on their personal and academic profile. It predicts a career field using a trained ML model on real-world student-like data, helping guide students in making informed academic and career decisions.


##  What This Project Does
- **Predicts Career Field** from 15+ student attributes
- **Applies Data Science Pipeline**: EDA → Feature Engineering → Model Training → Deployment
- Built with a **Machine Learning Model** (XGBoost Classifier)
- Fully interactive **Streamlit Web App**
- Designed for students, academic advisors, and educational counselors


## Folder Structure
academic-career-recommender-v2/
├── data/ # Datasets and processed features
├── models/ # Trained model and label encoders
├── scripts/ # Python scripts for EDA, model training, testing
├── dashboard/ # Streamlit frontend app
├── requirements.txt # Python dependencies
├── README.md # You're reading it now!
└── .gitignore # Files to be ignored by Git


## Why This is a Data Science & ML Project
This project follows a full **end-to-end data science workflow**, including:

- **Data Cleaning & Preprocessing**: Using `pandas` to clean and structure raw data
- **Feature Engineering**: Selecting meaningful features (e.g. GPA, Skills, Internships)
- **Label Encoding & Clustering**: Converting categorical labels to numeric for model training
- **Model Building**: Training an **XGBoost** multi-class classifier
- **Evaluation**: Testing the model and resolving feature-label mismatches
- **Deployment**: Creating an interactive **Streamlit dashboard** for real-time predictions


## 🧪 How to Run Locally
### 1. Clone this Repository

git clone https://github.com/your-username/academic-career-recommender-v2.git
cd academic-career-recommender-v2

### 2. Install the Requirements

pip install -r requirements.txt

### 3. Run the Streamlit App

streamlit run dashboard/app.py


## Tools and Technologies Used

Languages: Python
Libraries:
pandas, numpy – Data processing
xgboost, scikit-learn – Model training
pickle – Saving trained models
streamlit – Interactive dashboard
Version Control: Git + GitHub
IDE: VS Code


## Key Skills Demonstrated

-Machine Learning: Multi-class classification using XGBoost
-Data Science: Full pipeline from raw data to deployment
-Model Serialization: Save/load models with Pickle
-Frontend for ML: Real-time web UI with Streamlit
-Debugging & Troubleshooting: Feature mismatch resolution, label encoding errors, etc.


## Future Enhancements

-Add Top-K Career Suggestions (e.g., Top 3 paths)
-Visualize model explanations using SHAP
-Add user profile saving and authentication
-Export career recommendation reports


### Sample Inputs Used for Prediction

-GPA
-Internships
-Research Experience
-Communication Skills
-Teamwork
-Coding & Analytical Skills
-Leadership & Extracurriculars
-Field-Specific Courses
-Presentation & Networking Skills

