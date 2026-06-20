# HexSoftwares_Heart_Disease_Detection

## Project Overview
A machine learning model that predicts whether a patient has heart disease based on 13 health indicators.

## Results
- **Accuracy**: 83.61%
- **Algorithm**: Random Forest Classifier
- **Dataset**: UCI Heart Disease (303 patients)

## Features Used
| Feature | Description |
|---------|-------------|
| age | Age in years |
| sex | 1 = male, 0 = female |
| cp | Chest pain type (0-3) |
| trestbps | Resting blood pressure |
| chol | Cholesterol level |
| fbs | Fasting blood sugar |
| restecg | Resting ECG results |
| thalach | Max heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels |
| thal | Thalassemia |

## Top 5 Most Important Features
1. **oldpeak** - ST depression (most important)
2. **thalach** - Max heart rate
3. **ca** - Number of major vessels
4. **cp** - Chest pain type
5. **thal** - Thalassemia

## How to Run

### Option 1: Google Colab
1. Upload `heart.csv` and `heart_disease_detection.py` to Colab
2. Run the Python file

### Option 2: Local Machine
```bash
# Install dependencies
pip install -r requirements.txt

# Run the script
python heart_disease_detection.py
