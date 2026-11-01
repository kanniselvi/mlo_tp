
import os
import streamlit as st
import pandas as pd


python3 -m pip install joblib
import joblib
# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_package_model_v1.joblib")
model = joblib.load(model_path)


# Streamlit App

st.title("Tourism Package Purchase Prediction App")

st.write("""
This application predicts whether a customer is likely to purchase
a tourism package based on their demographic information,
travel preferences, and interaction details.
""")

# Customer Details

st.header("Customer Details")

Age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

TypeofContact = st.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Inquiry"]
)

CityTier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

Occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)

Gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

MaritalStatus = st.selectbox(
    "Marital Status",
    ["Married", "Unmarried", "Divorced"]
)

Designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

ProductPitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)


# Travel Details

st.header("Travel Details")

NumberOfPersonVisiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    max_value=10,
    value=2
)

PreferredPropertyStar = st.selectbox(
    "Preferred Property Star",
    [3, 4, 5]
)

NumberOfTrips = st.number_input(
    "Number of Trips",
    min_value=0,
    max_value=50,
    value=2
)

Passport = st.selectbox(
    "Passport",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

OwnCar = st.selectbox(
    "Own Car",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

NumberOfChildrenVisiting = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    max_value=10,
    value=0
)
# Interaction Details


st.header("Customer Interaction Details")

MonthlyIncome = st.number_input(
    "Monthly Income",
    min_value=0.0,
    max_value=1000000.0,
    value=25000.0,
    step=1000.0
)

PitchSatisfactionScore = st.selectbox(
    "Pitch Satisfaction Score",
    [1, 2, 3, 4, 5]
)

NumberOfFollowups = st.number_input(
    "Number of Followups",
    min_value=0,
    max_value=20,
    value=2
)

DurationOfPitch = st.number_input(
    "Duration of Pitch (minutes)",
    min_value=0,
    max_value=120,
    value=10
)

# Create Input DataFrame


input_data = pd.DataFrame([{

    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "Occupation": Occupation,
    "Gender": Gender,
    "MaritalStatus": MaritalStatus,
    "Designation": Designation,
    "ProductPitched": ProductPitched,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "PreferredPropertyStar": PreferredPropertyStar,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "MonthlyIncome": MonthlyIncome,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "NumberOfFollowups": NumberOfFollowups,
    "DurationOfPitch": DurationOfPitch

}])

# Prediction


if st.button("Predict Package Purchase"):

    prediction = model.predict(input_data)[0]

    result = (
        "Package Purchase"
        if prediction == 1
        else "No Package Purchase"
    )

    st.subheader("Prediction Result:")

    if prediction == 1:
        st.success(
            "The model predicts: **Customer is likely to purchase the package!**"
        )
    else:
        st.info(
            "The model predicts: **Customer is unlikely to purchase the package.**"
        )

    # Show probability if supported by the model
    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(input_data)[0][1]

        st.write(
            f"Purchase Probability: **{probability:.2%}**"
        )