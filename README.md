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

## 📈 Sample Visuals & Key Insights

<table>
  <tr>
    <td><img src="readmission_distribution.png" width="400"/></td>
    <td><img src="gender_vs_readmission.png" width="400"/></td>
  </tr>
  <tr>
    <td><b>🔍 Insight:</b> Around 30% of patients were readmitted within 30 days.</td>
    <td><b>🔍 Insight:</b> Slightly higher readmission observed in male patients.</td>
  </tr>
  <tr>
    <td><img src="age_vs_readmission.png" width="400"/></td>
    <td><img src="feature_importance.png" width="400"/></td>
  </tr>
  <tr>
    <td><b>🔍 Insight:</b> Older age groups tend to have higher readmission risk.</td>
    <td><b>🔍 Insight:</b> Comorbidity count, length of stay, and physician notes were most predictive.</td>
  </tr>
</table>

<div align="center">
  <img src="roc_curve.png" width="420"/>
  <p><b>🔍 Insight:</b> AUC of ~0.82 indicates strong model performance distinguishing readmissions.</p>
</div>

## 📁 Files Included
- `Healthcare_Readmission_Risk_Model.ipynb` – Complete Jupyter code
- `Healthcare_Readmission_Risk_Model.py` – Python script version
- `*.png` – Visual insights (charts)
- `readmission_model_output.csv` – Output for dashboarding

## 📎 Reference Project Link
**[Reference Project Link](https://github.com/vineethasusan/Healthcare-Readmission-Risk-Model)**
