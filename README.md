# 🏥 Healthcare Readmission Risk Model

![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Jupyter-orange)

This project predicts the risk of 30-day patient readmissions using clinical data and physician notes.

## 📌 Objective
- Predict unplanned readmission using ML (LogReg + Random Forest)
- Visualize feature importance for clinical interpretability
- Export risk predictions for Power BI dashboards

## 📊 Features Used
- Age, Gender, DiagnosisCode, ComorbidityCount
- Length of Stay, Discharge Type, Follow-Up Scheduled
- TF-IDF from `PhysicianNotes`

## 📈 Sample Visuals

<p float="left">
  <img src="readmission_distribution.png" width="400"/>
  <img src="gender_vs_readmission.png" width="400"/>
</p>

<p float="left">
  <img src="age_vs_readmission.png" width="400"/>
  <img src="feature_importance.png" width="400"/>
</p>

## 📁 Files Included
- `Healthcare_Readmission_Risk_Model.ipynb` – Complete Jupyter code
- `Healthcare_Readmission_Risk_Model.py` – Python script version
- `*.png` – Visual insights (charts)
- `readmission_model_output.csv` – Output for dashboarding

## 📎 Reference Project Link
**[Reference Project Link](https://github.com/vineethasusan/Healthcare-Readmission-Risk-Model)**
