import numpy as np
import pandas as pd
import streamlit as st
import pickle

# تنظیمات صفحه و آیکون پزشکی
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="centered"
)

# استایل CSS برای حس و حال مدرن بیمارستانی و پزشکی
st.markdown("""
<style>
    /* پس‌زمینه ملایم کلینیکی */
    .stApp {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 50%, #f8fafc 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* کارت‌های فرم */
    div[data-testid="stVerticalBlock"] > div:has(div.stRadio, div.stNumberInput) {
        background: rgba(255, 255, 255, 0.85);
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(14, 116, 144, 0.08);
        border: 1px solid #bae6fd;
        margin-bottom: 0.8rem;
    }

    /* دکمه پیش‌بینی */
    .stButton > button {
        background: linear-gradient(90deg, #0284c7, #0ea5e9);
        color: white;
        border: none;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
        padding: 0.6rem;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #0369a1, #0284c7);
        box-shadow: 0 6px 16px rgba(2, 132, 199, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# هدر و معرفی
st.markdown("<h1 style='text-align: center; color: #0369a1;'>🫀 Heart Disease Risk Assessment</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b;'>Clinical Decision Support System (10-Year CHD Risk)</p>", unsafe_allow_html=True)
st.divider()

# لود مدل‌ها
with open("model.pkl", "rb") as file:
    model = pickle.load(file)
with open("pca.pkl", "rb") as file:
    pca = pickle.load(file)

# دیکشنری‌ها
gender_dict = {"male": 1, "female": 0}
yes_no_dict = {"yes": 1, "no": 0}

# بخش ۱: سوابق و وضعیت اولیه (دو ستون)
st.subheader("📋 Patient History")
col1, col2 = st.columns(2)

with col1:
    gender_key = st.radio("Gender", list(gender_dict.keys()), horizontal=True)
    cigarett_key = st.radio("Smoking lately?", list(yes_no_dict.keys()), horizontal=True)
    blood_pressure_key = st.radio("Blood pressure medication?", list(yes_no_dict.keys()), horizontal=True)

with col2:
    prevalent_key = st.radio("History of stroke?", list(yes_no_dict.keys()), horizontal=True)
    high_blood_pressure_key = st.radio("High blood pressure history?", list(yes_no_dict.keys()), horizontal=True)
    diabets_key = st.radio("Diabetes?", list(yes_no_dict.keys()), horizontal=True)

gender = gender_dict[gender_key]
cigarett = yes_no_dict[cigarett_key]
blood_pressure = yes_no_dict[blood_pressure_key]
prevalent = yes_no_dict[prevalent_key]
high_blood_pressure = yes_no_dict[high_blood_pressure_key]
diabets = yes_no_dict[diabets_key]

# بخش ۲: معیارهای بالینی و علائم حیاتی
st.subheader("🩺 Vitals & Laboratory Values")
col3, col4 = st.columns(2)

with col3:
    age = st.number_input("Age (years)", min_value=42, max_value=70, step=1)
    cigg = st.number_input("Cigarettes per day", min_value=9, max_value=70, step=1)
    total = st.number_input("Total Cholesterol (mg/dL)", min_value=236, max_value=696, step=1)
    sysBP = st.number_input("Systolic Blood Pressure", min_value=132, max_value=295, step=1)

with col4:
    diaBP = st.number_input("Diastolic Blood Pressure", min_value=82, max_value=142, step=1)
    BMI = st.number_input("BMI", min_value=25, max_value=56, step=1)
    heartbeatsrate = st.number_input("Heart Rate (bpm)", min_value=75, max_value=143, step=1)
    glucose = st.number_input("Glucose level (mg/dL)", min_value=81, max_value=394, step=1)

st.write("")
if st.button("Analyze Risk"):
    data = np.array([[gender, age, cigarett, cigg, blood_pressure, prevalent, high_blood_pressure,
                      diabets, total, sysBP, diaBP, BMI, heartbeatsrate, glucose]])
    result = int(model.predict(data)[0])
    if result == 1:
        st.error("you suffer from CHD during 10 years")
    else:
        st.success("luckily you don't suffer from CHD during 10 years")