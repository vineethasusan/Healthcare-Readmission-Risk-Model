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

## 📈 Sample Visuals & Key Insights

<table>
  <tr>
    <td><img src="readmission_distribution.png" width="400"/></td>
    <td><img src="gender_vs_readmission.png" width="400"/></td>
  </tr>
  <tr>
    <td><b>🔍 Insight:</b> ~28–30% of patients were readmitted within 30 days (moderate class imbalance).</td>
    <td><b>🔍 Insight:</b> Female patients had more total admissions, but readmission rates are similar across genders.</td>
  </tr>
  <tr>
    <td><img src="age_vs_readmission.png" width="400"/></td>
    <td><img src="feature_importance.png" width="400"/></td>
  </tr>
  <tr>
    <td><b>🔍 Insight:</b> Median age is consistent across groups; readmitted patients have slightly broader age range.</td>
    <td><b>🔍 Insight:</b> Age is the strongest predictor, followed by length of stay and comorbidity count.</td>
  </tr>
</table>

<div align="center">
  <img src="roc_curve.png" width="420"/>
  <p><b>🔍 Insight:</b> Logistic Regression outperforms Random Forest here with AUC ~0.82 vs ~0.70.</p>
</div>

## 📁 Files Included
- `Healthcare_Readmission_Risk_Model.ipynb` – Complete Jupyter code
- `Healthcare_Readmission_Risk_Model.py` – Python script version
- `*.png` – Visual insights (charts)
- `readmission_model_output.csv` – Output for dashboarding

## 📎 Reference Project Link
**[Reference Project Link](https://github.com/vineethasusan/Healthcare-Readmission-Risk-Model)**
