# Mental Health Depression Risk Prediction

## Project Highlights

- Predicts depression risk using student mental‑health survey data.
- Full ML pipeline:
  - Data loading, cleaning, and exploratory data analysis.
  - Feature engineering and feature selection.
  - Model training, evaluation, and debiasing.
  - Deployment as a web app (Streamlit) and optional REST API (Flask).
- Final model:
  - Algorithm: Logistic Regression with class balancing.
  - Performance (approximate): F1 ≈ 0.86, Accuracy ≈ 0.84, ROC‑AUC ≈ 0.91.
  - Includes basic checks for class imbalance and bias (sensitivity vs specificity).

## Tech Stack

- Python 3.x  
- pandas, numpy  
- scikit‑learn, imbalanced‑learn  
- joblib  
- Streamlit (web app)  
 

## Project Structure

Key folders and files:

- `data`
  - `raw`
    - `mental-heath-in-tech-2016_20161114.csv`
    - `student_depression_dataset.csv`
  - `processed` – cleaned and transformed data.
- `notebooks`
  - `01_eda.ipynb` – Exploratory data analysis.
  - `02_data_cleaning.ipynb` – Cleaning and preprocessing.
  - `03_model_training.ipynb` – Model training and evaluation.
  - `04_report.ipynb` – Human‑readable project report.
 
- `app`
  - `app_streamlit.py` – Streamlit web app.
 
- `scripts/` – Helper scripts for data loading and preprocessing.
- `docs/` – Model comparison table, plots, and other artifacts.
- `README.md` – Project documentation.
- `requirements.txt` – Python dependencies.

## Setup and Installation

1. Clone the repository:

git clone https://github.com/<Surya97141>/Mental_health_Prediction.git
cd Mental_health_Prediction
