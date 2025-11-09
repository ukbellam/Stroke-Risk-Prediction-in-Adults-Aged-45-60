# Stroke-Risk-Prediction-in-Adults-Aged-45-60
This project applies Artificial Intelligence (AI) and Machine Learning (ML) techniques to predict stroke risk in adults aged 45–60 years. It uses PySpark for distributed data processing and modeling, helping clinicians identify high-risk individuals early and support preventive healthcare decisions.

This project applies Artificial Intelligence (AI) and Machine Learning (ML) techniques to predict stroke risk in adults aged 45–60 years.
It uses PySpark for distributed data processing and modeling, helping clinicians identify high-risk individuals early and support preventive healthcare decisions.

Project Overview
This project uses Artificial Intelligence and Machine Learning to predict stroke risk in adults aged 45–60.
It is built with PySpark for distributed data processing to help clinicians identify high-risk individuals early.
Dataset
Source: Kaggle Stroke Prediction Dataset
Records: 5,110 entries
Features: Age, Gender, Hypertension, Heart Disease, BMI, Glucose Level, and others
Target: stroke (1 = Yes, 0 = No)
Technologies and Tools
Python, PySpark, pandas, matplotlib, seaborn
Techniques: SMOTE, GridSearchCV, StandardScaler
Setup: Two Virtual Machines (Master–Worker)
Model Summary
Implemented a Random Forest Classifier in PySpark with probability calibration for reliability.
Top predictors were Age, Average Glucose Level, BMI, Hypertension, and Heart Disease.
The model demonstrated Spark’s efficiency for large-scale healthcare data analysis.
