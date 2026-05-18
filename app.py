from flask import Flask, render_template, request, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

import pickle
import numpy as np

# ================= INITIALIZE APP =================

app = Flask(__name__)

app.secret_key = "career_secret_key"

# ================= DATABASE CONFIG =================

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ================= USER MODEL =================

class User(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    email = db.Column(
        db.String(100),
        unique=True,
        nullable=False
    )

    password = db.Column(
        db.String(200),
        nullable=False
    )

# Create database
with app.app_context():
    db.create_all()

# ================= LOAD MODEL FILES =================

scaler = pickle.load(open("Models/scaler.pkl", "rb"))
model = pickle.load(open("Models/model.pkl", "rb"))

# ================= CAREER CLASSES =================

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

    gender_encoded = 1 if gender.lower() == 'female' else 0
    part_time_job_encoded = 1 if part_time_job else 0
    extracurricular_encoded = 1 if extracurricular_activities else 0

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

    scaled_features = scaler.transform(features)

    probabilities = model.predict_proba(scaled_features)

    top_indices = np.argsort(-probabilities[0])[:3]

    results = [
        (class_names[i], round(probabilities[0][i] * 100, 2))
        for i in top_indices
    ]

    return results

# ================= ROUTES =================

@app.route('/')
def home():
    return render_template('home.html')

# ================= SIGNUP =================

@app.route('/signup', methods=['GET', 'POST'])
def signup():

    if request.method == 'POST':

        email = request.form['email']
        password = request.form['password']

        existing_user = User.query.filter_by(email=email).first()

        if existing_user:
            return "User already exists!"

        hashed_password = generate_password_hash(password)

        new_user = User(
            email=email,
            password=hashed_password
        )

        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for('login'))

    return render_template('signup.html')

# ================= LOGIN =================

@app.route('/login', methods=['GET', 'POST'])
def login():

    if request.method == 'POST':

        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):

            session['user'] = user.email

            return redirect(url_for('recommend'))

        else:
            return "Invalid Email or Password"

    return render_template('login.html')

# ================= LOGOUT =================

@app.route('/logout')
def logout():

    session.pop('user', None)

    return redirect(url_for('home'))

# ================= RECOMMEND PAGE =================

@app.route('/recommend')
def recommend():

    if 'user' not in session:
        return redirect(url_for('login'))

    return render_template('recommend.html')

# ================= PREDICTION =================

@app.route('/pred', methods=['POST'])
def pred():

    if 'user' not in session:
        return redirect(url_for('login'))

    try:

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

        recommendations = Recommendations(

            gender,
            part_time_job,
            absence_days,
            extracurricular,
            study_hours,
            math,
            history,
            physics,
            chemistry,
            biology,
            english,
            geography,
            total,
            average

        )

        return render_template(
            'results.html',
            recommendations=recommendations
        )

    except Exception as e:
        return f"Error: {e}"

# ================= RUN APP =================

if __name__ == "__main__":
    app.run(debug=True)