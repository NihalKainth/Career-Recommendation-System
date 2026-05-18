import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("student-scores.csv")

# Encode categorical columns
df['gender'] = df['gender'].map({'male': 0, 'female': 1})
df['part_time_job'] = df['part_time_job'].map({False: 0, True: 1})
df['extracurricular_activities'] = df['extracurricular_activities'].map({False: 0, True: 1})

# Features and target
X = df.drop('career', axis=1)   # Target column
y = df['career']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# Create Models folder
os.makedirs("Models", exist_ok=True)

# Save model and scaler
pickle.dump(model, open("Models/model.pkl", "wb"))
pickle.dump(scaler, open("Models/scaler.pkl", "wb"))

print("✅ Model and scaler saved successfully!")