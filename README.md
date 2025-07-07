# NU-Challenge-Explainability-In-Ai-Models
# 🔍 Bias Detection and Explainability in a Job Screening AI Model

This project was developed for a 48-hour internship challenge focused on uncovering bias and improving transparency in AI-driven hiring decisions. We investigate fairness in a **text-based job screening model** using techniques like BERT-based classification, SHAP/LIME explainability, and counterfactual data augmentation.

---

## 📄 Challenge Overview

- **Task:** Detect gender bias in an AI model that predicts hiring decisions from resume-style text.
- **Approach:** 
  - Generate synthetic resume text from structured features.
  - Train a binary classifier (Hire vs. No Hire).
  - Evaluate model fairness across gender groups.
  - Apply SHAP and LIME to explain predictions.
  - Mitigate bias via counterfactual augmentation (gender swapping).

---

## 🧠 Model Pipeline

- **Data Generation:** Convert tabular features (age, gender, education, etc.) into `Resume_Text`.
- **Model:** Fine-tuned `bert-base-uncased` using HuggingFace Transformers.
- **Evaluation Metrics:**
  - Accuracy
  - Demographic Parity
  - Equal Opportunity
  - Average Odds Difference

---

## 📊 Key Results

| Metric                     | Original Model | After Mitigation |
|---------------------------|----------------|------------------|
| Accuracy                  | 89.6%          | 92%              |
| Demographic Parity Gap    | 0.0023         | 0.0062           |
| Equal Opportunity Gap     | 0.0023         | 0.035            |
| Average Odds Difference   | 0.0257         | 0.022            |

✅ Bias mitigation via counterfactual data augmentation led to **improved accuracy** and **balanced fairness metrics**, with only a small tradeoff in equal opportunity.

---

## 🛠️ Tools & Libraries

- Python, Pandas, NumPy
- HuggingFace Transformers
- SHAP, LIME
- Scikit-learn, Matplotlib, Seaborn

---

## 📂 Project Structure

├── data/
│ └── resume_dataset.csv
├── notebook/
│ └── bias_detection_and_explainability.ipynb
├── README.md
└── report.pdf


# Run the notebook
jupyter notebook notebook/Bias_Detection_And_Explainability_In_Ai_Models_KareemWael.ipynb
