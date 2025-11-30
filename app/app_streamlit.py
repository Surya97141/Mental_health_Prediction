import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


st.set_page_config(
    page_title="Depression Risk Assessment",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model_data():
    try:
        model = joblib.load('models/model_best.pkl')
        scaler = joblib.load('models/scaler.pkl')
        with open('models/selected_features.json') as f:
            features = json.load(f)
        return model, scaler, features, True
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None, False

model, scaler, features, model_loaded = load_model_data()


st.title(" Mental Health Depression Risk Assessment")

st.markdown("""
This tool uses a debiased machine learning model to assess depression risk based on several mental health indicators.  
It is **only a screening tool** and must **not** be used as a substitute for professional medical diagnosis.
""")

if not model_loaded:
    st.error("Unable to load model files. Please ensure all required files exist in the `models/` directory.")
    st.stop()


st.sidebar.header(" Assessment Form")
st.sidebar.markdown("Provide the following information about the individual:")
st.sidebar.markdown("---")

input_data = {}

for feature in features:
    input_data[feature] = st.sidebar.number_input(
        label=f"{feature}",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=1.0,
        help=f"Enter value for {feature}"
    )

st.sidebar.markdown("---")


col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(" Model Performance (Debiased Model)")

    # NOTE: Replace these values with your actual metrics from results_df
    metrics_data = pd.DataFrame({
        "Metric":    ["Algorithm",                    "F1 Score", "Accuracy", "Precision", "Recall", "ROC-AUC"],
        "Value":     ["Logistic Regression (Balanced)", "0.86",    "0.84",    "0.85",      "0.87",   "0.91"]
    })
    st.dataframe(metrics_data, use_container_width=True, hide_index=True)

with col2:
    st.subheader(" Input Features Used")
    st.write(f"Total number of features: **{len(features)}**")
    st.write("")
    for i, feat in enumerate(features[:5], 1):
        st.write(f"{i}. {feat}")
    if len(features) > 5:
        st.write(f"... plus **{len(features) - 5}** additional features")

st.markdown("---")


st.subheader("🔮 Generate Depression Risk Assessment")

if st.button("Run Assessment", use_container_width=True, type="primary"):
    try:
        # Prepare input
        input_df = pd.DataFrame([input_data])

        # Scale with same scaler used in training
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]

        st.success("Assessment completed successfully.")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            if prediction == 1:
                st.error(" HIGH RISK")
                st.write("The model indicates an **elevated risk of depression**.")
            else:
                st.success(" LOW RISK")
                st.write("The model indicates a **low risk of depression**.")

        with result_col2:
            st.metric(
                "Model Confidence",
                f"{max(probabilities) * 100:.1f}%"
            )

        with result_col3:
            st.metric(
                "Depression Probability",
                f"{probabilities[1]:.4f}"
            )

        # Detailed probability breakdown
        st.subheader(" Probability Analysis")

        prob_details = pd.DataFrame({
            "Classification": ["Not Depressed", "Depressed"],
            "Probability":    [probabilities[0], probabilities[1]],
            "Percentage":     [f"{probabilities[0]*100:.2f}%", f"{probabilities[1]*100:.2f}%"]
        })
        st.dataframe(prob_details, use_container_width=True, hide_index=True)

        # Visual risk distribution
        st.subheader(" Risk Distribution")
        chart_data = pd.DataFrame({
            "Classification": ["Not Depressed", "Depressed"],
            "Probability":    [probabilities[0], probabilities[1]]
        }).set_index("Classification")
        st.bar_chart(chart_data)

        # Recommendations
        st.subheader(" Recommended Next Steps")
        if prediction == 1:
            st.warning("""
**High Risk Assessment**

Based on this screening, the model suggests a higher likelihood of depression.  
Consider the following actions:

- Consult a licensed mental health professional for a full evaluation  
- Discuss your concerns with your primary care physician  
- Seek counseling or therapy support  
- Talk openly with trusted friends or family members  
- If you have thoughts of self‑harm or suicide, seek **immediate** help via emergency services or crisis hotlines
""")
        else:
            st.info("""
**Low Risk Assessment**

To maintain and support good mental health:

- Maintain regular physical activity and exercise  
- Practice stress‑management and relaxation techniques  
- Maintain healthy sleep habits (7–9 hours per night)  
- Keep strong social connections with friends and family  
- Monitor your mental health and seek professional help if concerns arise in the future
""")

    except Exception as e:
        st.error(f"An error occurred during the assessment: {e}")
        st.write("Please verify your inputs and try again.")

st.markdown("---")

# ===================== DISCLAIMER =====================
st.info("""
###  Important Medical Disclaimer

This tool is **not a diagnostic instrument**.  
It is a machine‑learning‑based screening aid trained on historical survey data.

If you believe you may be experiencing depression or any mental health concern:

- Consult a licensed mental health professional for proper diagnosis and treatment  
- Contact a local or national suicide prevention or crisis hotline  
- Seek emergency medical care if you feel you are in immediate danger  

**Do not** make major medical or life decisions based solely on this tool.
""")