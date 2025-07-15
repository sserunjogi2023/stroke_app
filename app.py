
import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model
import warnings

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile
import base64
import os

import base64  # Ensure this is imported


warnings.filterwarnings("ignore")  # Suppress warnings that cause the feature_weights error

model = load_model('xgboost_pycaret_stroke_model')

def generate_pdf(data):
    import tempfile
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(temp_file.name, pagesize=letter)
    c.drawString(100, 750, "Stroke Prediction Report")

    for i, row in data.iterrows():
        line = f"{i+1}. Gender: {row['gender']}, Age: {row['age']}, Risk: {row['prediction_label']}, Score: {row['prediction_score']:.2f}"
        c.drawString(50, 700 - (i * 20), line)
        if i > 20:
            break

    c.save()
    return temp_file.name


st.title("Stroke Risk Prediction App")

gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.slider("Age", 1, 100, 30)
hypertension = st.radio("Hypertension", [0, 1])
heart_disease = st.radio("Heart Disease", [0, 1])
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", value=90.0)
bmi = st.number_input("BMI", value=25.0)
smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

if st.button("Predict"):
    input_data = pd.DataFrame([{
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }])
    pred = predict_model(estimator=model, data=input_data)
    label = pred['prediction_label'][0]
    score = pred['prediction_score'][0]

    if label == 0:
        st.success(f"No Stroke Risk (Confidence: {score:.2f})")
    else:
        st.error(f"Stroke Risk Detected (Confidence: {score:.2f})")

st.header("📂 Batch Prediction from CSV")

uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.error("Uploaded CSV file is empty! Please upload a valid CSV.")
        else:
            prediction_results = predict_model(model, data=df)

            # Show uploaded data preview
            st.subheader("📋 Uploaded Data Preview")
            st.dataframe(df.head())

            # Show prediction results
            st.subheader("🧠 Prediction Results")
            st.dataframe(prediction_results[['prediction_label', 'prediction_score']])

            # Generate PDF report
            pdf_path = generate_pdf(prediction_results)
            with open(pdf_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode("utf-8")
            href = f'<a href="data:application/pdf;base64,{base64_pdf}" download="stroke_report.pdf">📄 Download PDF Report</a>'
            st.markdown(href, unsafe_allow_html=True)

            # Doctor summary
            st.subheader("Doctor Summary")
            high_risk_count = prediction_results[prediction_results['prediction_label'] == 1].shape[0]
            low_risk_count = prediction_results[prediction_results['prediction_label'] == 0].shape[0]
            st.info(f"Total High Risk Patients: {high_risk_count}")
            st.info(f"Total Low Risk Patients: {low_risk_count}")

            # Filtering
            st.subheader("Filter by Stroke Risk")
            risk_filter = st.radio("Show patients with:", ["All", "High Risk", "Low Risk"])
            if risk_filter == "High Risk":
                filtered_data = prediction_results[prediction_results['prediction_label'] == 1]
            elif risk_filter == "Low Risk":
                filtered_data = prediction_results[prediction_results['prediction_label'] == 0]
            else:
                filtered_data = prediction_results

            st.dataframe(filtered_data)

            # Download CSV report
            final_results = pd.concat([df, prediction_results[['prediction_label', 'prediction_score']]], axis=1)
            csv = final_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Prediction Report as CSV",
                data=csv,
                file_name='stroke_predictions_report.csv',
                mime='text/csv'
            )

            # Visualization
            st.subheader("Age Distribution")
            st.bar_chart(df['age'].value_counts().sort_index())

            st.subheader("Predicted Stroke Risk Count")
            st.bar_chart(prediction_results['prediction_label'].value_counts())

    except Exception as e:
        st.error(f"Error processing the file: {e}")

else:
    st.info("Please upload a CSV file to get predictions, PDF report, and filtering options.")


