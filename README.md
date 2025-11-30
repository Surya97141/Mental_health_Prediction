# Mental Health Depression Risk Prediction

This project builds an end‑to‑end machine learning system that estimates depression risk for students based on survey responses. It includes data preprocessing, model training with basic debiasing, and deployment as an interactive Streamlit web app.

> **Important:** This is a screening and educational tool only. It must **not** be used as a substitute for professional medical diagnosis or treatment.

---

## 1. Project Highlights

- Predicts depression risk using student mental‑health survey data.
- Complete ML pipeline:
  - Data loading, cleaning, and exploratory data analysis.
  - Feature engineering and feature selection.
  - Model training, evaluation, and debiasing.
  - Deployment as a web app with Streamlit.
- Final model:
  - Algorithm: Logistic Regression with `class_weight='balanced'`.
  - Approximate performance (you can update with exact numbers):
    - F1 Score ≈ 0.86  
    - Accuracy ≈ 0.84  
    - ROC‑AUC ≈ 0.91  
  - Includes checks for class imbalance and basic bias (sensitivity vs specificity).

---

## 2. Tech Stack

- Python 3.x  
- pandas, numpy  
- scikit‑learn, imbalanced‑learn  
- joblib  
- Streamlit  

---

## 3. Project Structure

Mental_health_Prediction/
│
├─ data/
│ ├─ raw/
│ │ ├─ student_depression_dataset.csv
│ │ └─ mental-heath-in-tech-2016_20161114.csv
│ └─ processed/ # cleaned / intermediate files (optional)
│
├─ models/
│ ├─ model_best.pkl # final trained (debiased) model
│ ├─ scaler.pkl # StandardScaler used during training
│ └─ selected_features.json # list of feature names used by the model
│
├─ notebooks/
│ ├─ 01_eda.ipynb # exploratory data analysis
│ ├─ 02_data_cleaning.ipynb # preprocessing & cleaning
│ ├─ 03_model_training.ipynb # model training & debiasing
│ └─ 04_report.ipynb # final human‑readable report
│
├─ app_streamlit.py # Streamlit web application
├─ requirements.txt # Python dependencies
└─ README.md # this file

text

---

## 4. Setup and Installation

1. **Clone the repository**

git clone https://github.com/<your-username>/Mental_health_Prediction.git
cd Mental_health_Prediction

text

2. **Create and activate a virtual environment** (recommended)

python -m venv venv
venv\Scripts\activate # On Windows

source venv/bin/activate # On macOS / Linux
text

3. **Install dependencies**

pip install -r requirements.txt

text

4. **(Optional) Regenerate model files**

If `models/model_best.pkl`, `models/scaler.pkl`, or `models/selected_features.json` are missing, run the training notebook:

- Open `notebooks/03_model_training.ipynb`  
- Run all cells to retrain and save the model and scaler.

---

## 5. Running the Streamlit Web App

From the project root:

streamlit run app_streamlit.py

text

Then open the URL shown in the terminal (usually `http://localhost:8501`).

The app will:

- Ask for input values for key features (e.g., academic pressure, stress, sleep, satisfaction).
- Show a prediction:
  - **HIGH RISK** or **LOW RISK** of depression.
  - Probabilities for “Not Depressed” vs “Depressed”.
- Display a simple probability chart and suggested next steps.
- Show a clear medical disclaimer.

---

## 6. Methodology (Short Summary)

1. **Data Cleaning**
   - Dropped rows with missing target.
   - Imputed numeric features with median and categorical with most frequent value.
   - Encoded categorical variables.

2. **Feature Engineering & Selection**
   - Created simple interaction features between important numeric columns.
   - Trained a small Random Forest to compute feature importances.
   - Selected the top ~15 features for the final model.

3. **Model Training & Debiasing**
   - Trained multiple models and chose Logistic Regression.
   - Handled class imbalance with `class_weight='balanced'` (and optionally SMOTE in training).
   - Evaluated metrics including F1, accuracy, precision, recall, ROC‑AUC, sensitivity, specificity, and bias gap.
   - Performed a “healthy profile” sanity check to ensure obviously low‑risk inputs are not always predicted as depressed.

---

## 7. Limitations & Ethical Considerations

- Based on self‑reported survey data; answers may be noisy or biased.
- Dataset may not fully represent all populations or contexts.
- The model is a **screening aid**, not a diagnostic tool.
- Predictions must always be interpreted by qualified professionals and not be the sole basis for serious decisions.

---

## 8. Acknowledgements

- Student depression / mental‑health datasets (publicly available).
- Open‑source libraries: pandas, numpy, scikit‑learn, imbalanced‑learn, joblib, Streamlit.
