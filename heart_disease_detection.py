
"""
Heart Disease Detection
Hex Softwares Internship Task 2
Accuracy: 83.61%
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
df = pd.read_csv('heart.csv')

print("="*50)
print("   HEART DISEASE DETECTION")
print("   Hex Softwares Internship")
print("="*50)

print(f"\nDataset shape: {df.shape[0]} patients, {df.shape[1]} columns")

# Target distribution
print("\nTarget Distribution:")
print(df['target'].value_counts())

# Split data
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Disease', 'Disease']))

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 Most Important Features:")
print(feature_importance.head(5))

# Prediction function
def predict_heart_disease(patient_data):
    import pandas as pd
    patient_df = pd.DataFrame([patient_data], columns=X.columns)
    prediction = model.predict(patient_df)[0]
    prob = model.predict_proba(patient_df)[0][1]
    
    if prediction == 1:
        return f"High Risk ({prob*100:.1f}% confidence)"
    else:
        return f"Low Risk ({prob*100:.1f}% confidence)"

# Test
patient = [63, 1, 3, 145, 233, 0, 0, 150, 0, 2.3, 0, 0, 1]
result = predict_heart_disease(patient)
print(f"\nSample Prediction: {result}")

print("\n" + "="*50)
print("   PROJECT COMPLETE!")
print("   Hex Softwares")
print("="*50)
