import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# -----------------------------
# Load trained pipeline from HF
# -----------------------------
# Update repo_id/filename if you changed them during training
MODEL_REPO = "Yashwanthsairam/tourism-prediction-package"
MODEL_FILE = "best_tourism_wellness_model_v1.joblib"

model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
model = joblib.load(model_path)   # full sklearn Pipeline (preprocess + XGBoost)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Wellness Tourism Purchase Predictor")
st.write("""
Predict whether a customer is likely to purchase the **Wellness Tourism Package** based on their profile and interaction data.
Fill in the details below and click **Predict**.
""")

st.markdown("### Customer Details")
col1, col2, col3 = st.columns(3)

with col1:
    Age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
    NumberOfPersonVisiting = st.number_input("No. of Persons Visiting", min_value=0, max_value=20, value=2, step=1)
    PreferredPropertyStar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5], index=2)
    NumberOfTrips = st.number_input("Avg Trips per Year", min_value=0, max_value=50, value=2, step=1)

with col2:
    NumberOfChildrenVisiting = st.number_input("Children Visiting (under 5)", min_value=0, max_value=10, value=0, step=1)
    MonthlyIncome = st.number_input("Monthly Income", min_value=0, max_value=10_000_000, value=50_000, step=1000)
    Passport = st.selectbox("Has Passport?", [0, 1], index=1)
    OwnCar = st.selectbox("Owns Car?", [0, 1], index=0)

with col3:
    CityTier = st.selectbox("City Tier", [1, 2, 3], index=0)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    Occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Business", "Large Business", "Other"])

st.markdown("### Interaction Data")
col4, col5, col6 = st.columns(3)

with col4:
    TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])

with col5:
    ProductPitched = st.selectbox(
        "Product Pitched",
        ["Basic", "Deluxe", "Elite", "King", "Standard", "Special Offer"]
    )

with col6:
    Designation = st.selectbox(
        "Designation",
        ["Executive", "Manager", "Senior Manager", "AVP", "VP", "Other"]
    )

PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3, step=1)
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=50, value=2, step=1)
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=300, value=30, step=1)

# Assemble input row (columns must match training features; the pipeline handles encoding/scaling)
input_row = pd.DataFrame([{
    "Age": Age,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "PreferredPropertyStar": PreferredPropertyStar,
    "NumberOfTrips": NumberOfTrips,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "MonthlyIncome": MonthlyIncome,
    "Passport": Passport,
    "OwnCar": OwnCar,
    "CityTier": CityTier,
    "Gender": Gender,
    "MaritalStatus": MaritalStatus,
    "Occupation": Occupation,
    "TypeofContact": TypeofContact,
    "ProductPitched": ProductPitched,
    "Designation": Designation,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "NumberOfFollowups": NumberOfFollowups,
    "DurationOfPitch": DurationOfPitch
}])

st.markdown("---")
if st.button("Predict Purchase Likelihood"):
    # The pipeline includes preprocessing, so we can call predict directly.
    # If you want probability & a custom threshold, use predict_proba and compare.
    pred = model.predict(input_row)[0]
    proba = None
    try:
        proba = model.predict_proba(input_row)[0][1]
    except Exception:
        pass

    st.subheader("Prediction Result")
    label = "Will Purchase (1)" if pred == 1 else "Will Not Purchase (0)"
    if proba is not None:
        st.success(f"**{label}**  —  Confidence (P=1): **{proba:.3f}**")
    else:
        st.success(f"**{label}**")
