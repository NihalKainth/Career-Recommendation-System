# ================= INSTALL DEPENDENCIES =================
# pip install scikit-learn==1.3.2
# pip install numpy
# pip install flask

# ================= IMPORT PACKAGES =================
from flask import Flask, render_template, request
import pickle
import numpy as np

# ================= INITIALIZE APP =================
app = Flask(__name__)

# ================= LOAD MODEL FILES =================
scaler = pickle.load(open("Models/scaler.pkl", "rb"))
model = pickle.load(open("Models/model.pkl", "rb"))

# Career classes
class_names = [
    'Lawyer', 'Doctor', 'Government Officer', 'Artist', 'Unknown',
    'Software Engineer', 'Teacher', 'Business Owner', 'Scientist',
    'Banker', 'Writer', 'Accountant', 'Designer',
    'Construction Engineer', 'Game Developer', 'Stock Investor',
    'Real Estate Developer'
]

# ================= RECOMMENDATION FUNCTION =================
def Recommendations(gender, part_time_job, absence_days, extracurricular_activities,
                    weekly_self_study_hours, math_score, history_score, physics_score,
                    chemistry_score, biology_score, english_score, geography_score,
                    total_score, average_score):

    # Encode categorical variables
    gender_encoded = 1 if gender.lower() == 'female' else 0
    part_time_job_encoded = 1 if part_time_job else 0
    extracurricular_encoded = 1 if extracurricular_activities else 0

    # Create feature array
    features = np.array([[
        gender_encoded,
        part_time_job_encoded,
        absence_days,
        extracurricular_encoded,
        weekly_self_study_hours,
        math_score,
        history_score,
        physics_score,
        chemistry_score,
        biology_score,
        english_score,
        geography_score,
        total_score,
        average_score
    ]])

    # Scale features
    scaled_features = scaler.transform(features)

    # Predict probabilities
    probabilities = model.predict_proba(scaled_features)

    # Get top 3 predictions
    top_indices = np.argsort(-probabilities[0])[:3]

    results = [(class_names[i], round(probabilities[0][i] * 100, 2)) for i in top_indices]

    return results


# ================= ROUTES =================

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Simply redirect to assessment or simulate login for now
        return render_template('recommend.html')
    return render_template('login.html')


@app.route('/recommend')
def recommend():
    return render_template('recommend.html')


@app.route('/pred', methods=['POST'])
def pred():
    try:
        # Get form data
        gender = request.form['gender']
        part_time_job = request.form['part_time_job'] == 'true'
        absence_days = int(request.form['absence_days'])
        extracurricular = request.form['extracurricular_activities'] == 'true'
        study_hours = int(request.form['weekly_self_study_hours'])

        math = int(request.form['math_score'])
        history = int(request.form['history_score'])
        physics = int(request.form['physics_score'])
        chemistry = int(request.form['chemistry_score'])
        biology = int(request.form['biology_score'])
        english = int(request.form['english_score'])
        geography = int(request.form['geography_score'])

        total = float(request.form['total_score'])
        average = float(request.form['average_score'])

        # Get predictions
        recommendations = Recommendations(
            gender, part_time_job, absence_days, extracurricular,
            study_hours, math, history, physics,
            chemistry, biology, english, geography,
            total, average
        )

        return render_template('results.html', recommendations=recommendations)

    except Exception as e:
        return f"Error: {e}"


# ================= RUN APP =================
if __name__ == "__main__":
    app.run(debug=True)