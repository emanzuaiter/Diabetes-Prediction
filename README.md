# 🩺 Early Diabetes Risk Prediction Pipeline

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458)
![Milestone](https://img.shields.io/badge/Project-Milestone%201-brightgreen)

## 📌 1. Project Overview & Business Problem
* **Clinical Context:** Type-2 diabetes frequently develops asymptomatically, leading to delayed clinical diagnoses, preventable chronic complications (e.g., neuropathy, cardiovascular disease), and high long-term treatment costs.
* **Business Problem:** Outpatient healthcare centers need an automated preliminary screening workflow to flag high-risk patients during routine consultations prior to ordering extensive or invasive laboratory panels.
* **Core Answerable Question:** Can standard non-invasive physiological vitals (e.g., glucose, blood pressure, BMI, age) reliably predict whether a patient is diabetic (`Outcome: 0 or 1`) to enable early preventative intervention?

---

## 🎯 2. Objectives & Success Metrics
* **Core Objective:** Build, evaluate, and benchmark a binary classification pipeline to predict diabetes onset from tabular patient health indicators.
* **Target Success Criteria:**
  * **Recall / Sensitivity ≥ 0.80:** Prioritizing the detection of positive cases to minimize dangerous false negatives in a medical setting.
  * **Accuracy ≥ 0.60:** Maintaining a reliable baseline diagnostic performance.
  * **ROC-AUC > 0.82:** Ensuring strong diagnostic discrimination across multiple clinical decision thresholds.
---

## 📊 3. Dataset Specification
* **Source:** Kaggle — [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/hasibur013/diabetes-dataset)
* **Target Variable:** `Outcome` (Binary: `0` = Non-diabetic, `1` = Diabetic)
* **Dataset Shape:** 768 rows × 9 columns (8 predictor features + 1 target column)

### Data Dictionary
| Feature Name | Data Type | Description |
| :--- | :--- | :--- |
| `Pregnancies` | Integer | Number of times pregnant |
| `Glucose` | Integer | Plasma glucose concentration at 2 hours in an oral glucose tolerance test |
| `BloodPressure` | Integer | Diastolic blood pressure ($mm\ Hg$) |
| `SkinThickness` | Integer | Triceps skin fold thickness ($mm$) |
| `Insulin` | Integer | 2-Hour serum insulin ($\mu U/ml$) |
| `BMI` | Float | Body mass index ($weight\ in\ kg / (height\ in\ m)^2$) |
| `DiabetesPedigreeFunction` | Float | Diabetes pedigree function (genetic risk score) |
| `Age` | Integer | Age in years |
| **`Outcome`** | **Integer (Target)** | **Diagnostic status (0 = Negative, 1 = Positive)** |

---

## 🧬 4. Data Augmentation via CTGAN
To address the limited cohort size (768 records) and improve model generalization, we applied **Generative Data Augmentation** using **CTGAN (Conditional Tabular GAN)**:

* **Why CTGAN?** Tabular clinical data often features non-Gaussian, multi-modal distributions that traditional oversampling methods (like SMOTE) can distort through linear interpolation. CTGAN uses deep adversarial learning with mode-specific normalization to capture non-linear feature relationships while avoiding physiologically impossible combinations.
* **Epidemiological Ratio Preservation:** CTGAN was conditioned to preserve the natural ~35% positive diabetes prevalence (rather than forcing an artificial 50/50 balance), protecting against base-rate fallacy and preventing excessive false-positive alarms.
* **Zero-Leakage Training:** CTGAN was fitted strictly on the training partition so that validation and test sets remained completely unexposed.
* **Physical Bound Clipping:** Generated synthetic samples were bounded to empirical medical ranges to eliminate tail hallucinations.
* **Fidelity Validation:** Statistical audits (including Kolmogorov-Smirnov tests and correlation matrix comparisons) confirmed high distribution fidelity to real patient data with zero row memorization. 


