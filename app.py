import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Student Depression Predictor",
    page_icon="🧠",
    layout="wide"
)

# ----------------------------
# CUSTOM STYLE (UI IMPROVEMENT)
# ----------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 10px;
        padding: 10px;
    }
    .stSlider label, .stNumberInput label {
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# LOAD MODEL
# ----------------------------
model = joblib.load("student_depression_model.pkl")

# ----------------------------
# HEADER
# ----------------------------
st.title("🧠 Student Depression Prediction System")
st.markdown("### 🚀 AI-powered Mental Health Risk Detection")

st.write("---")

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.header("📊 About")
st.sidebar.info(
    """
    This app uses Machine Learning (XGBoost)  
    to predict depression risk based on student lifestyle.
    
    ✔ Academic Pressure  
    ✔ Sleep Duration  
    ✔ Financial Stress  
    ✔ Study Hours  
    """
)

# ----------------------------
# INPUT SECTION (CARD STYLE)
# ----------------------------
st.subheader("📋 Enter Student Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 10, 60, 20)
    cgpa = st.number_input("CGPA", 0.0, 4.0, 3.0)

with col2:
    academic_pressure = st.slider("Academic Pressure", 0, 10, 5)
    financial_stress = st.slider("Financial Stress", 0, 10, 5)

with col3:
    work_study_hours = st.slider("Work/Study Hours", 0, 16, 4)
    sleep_duration = st.selectbox(
        "Sleep Duration",
        ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"]
    )

st.write("---")

# ----------------------------
# CREATE INPUT DATA
# ----------------------------
input_data = pd.DataFrame([{
    "Age": age,
    "Academic Pressure": academic_pressure,
    "CGPA": cgpa,
    "Work/Study Hours": work_study_hours,
    "Financial Stress": financial_stress,
    "Sleep Duration": sleep_duration
}])

# ----------------------------
# PREDICTION BUTTON
# ----------------------------
if st.button("🔍 Analyze Depression Risk"):

    prediction = model.predict(input_data)[0]

    try:
        probability = model.predict_proba(input_data)[0][1]
    except:
        probability = None

    st.write("---")

    # ----------------------------
    # RESULT DISPLAY (BIG STYLE)
    # ----------------------------
    if prediction == 1:
        st.error("⚠ High Risk of Depression")
    else:
        st.success("✅ Low Risk of Depression")

    # ----------------------------
    # PROBABILITY BAR
    # ----------------------------
    if probability is not None:
        st.subheader("📊 Prediction Confidence")
        st.progress(int(probability * 100))
        st.write(f"Confidence: {probability*100:.2f}%")

    # ----------------------------
    # FEATURE IMPORTANCE
    # ----------------------------
    try:
        st.write("---")
        st.subheader("📈 Key Influencing Factors")

        model_step = model.named_steps['model']
        preprocessor = model.named_steps['preprocessor']

        feature_names = preprocessor.get_feature_names_out()
        importances = model_step.feature_importances_

        feat_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        fig, ax = plt.subplots()
        ax.barh(feat_df["Feature"][:8], feat_df["Importance"][:8])
        ax.invert_yaxis()

        st.pyplot(fig)

    except:
        st.warning("Feature importance not available.")

# ----------------------------
# FOOTER
# ----------------------------
st.write("---")
st.markdown(
    "<center>Built with ❤️ using Machine Learning + Streamlit | By Uthpala 🚀</center>",
    unsafe_allow_html=True
)