import streamlit as st
import pandas as pd
import requests

st.title("Machine Learning API Frontend")

# Section pour entraîner le modèle
st.header("Train the Model")
data_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if data_file is not None:
    df = pd.read_csv(data_file)
    st.write("Dataset Preview:")
    st.write(df.head())
    
    target = st.selectbox("Select target column", df.columns)
    
    if st.button("Train Model"):
        data_json = df.to_dict(orient='records')
        response = requests.post("http://localhost:8000/training", json={"data": data_json, "target": target})
        if response.status_code == 200:
            st.success("Model trained successfully")
            st.write(response.json())
        else:
            st.error("Error in training model")
            st.write(response.json())

# Section pour faire des prédictions
st.header("Predict")
with st.form("prediction_form"):
    Age = st.number_input("Age", step=1)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Income = st.number_input("Income")
    VisitFrequency = st.selectbox("VisitFrequency", ["Daily", "Weekly", "Monthly", "Rarely"])
    AverageSpend = st.number_input("AverageSpend")
    PreferredCuisine = st.selectbox("PreferredCuisine", ["Indian", "Chinese", "American", "Mexican", "Italian"])
    TimeOfVisit = st.selectbox("TimeOfVisit", ["Breakfast", "Lunch", "Dinner"])
    GroupSize = st.number_input("GroupSize", step=1)
    DiningOccasion = st.selectbox("DiningOccasion", ["Casual", "Business", "Formal"])
    MealType = st.selectbox("MealType", ["Dine-in", "Takeaway"])
    OnlineReservation = st.selectbox("OnlineReservation", [0, 1])
    DeliveryOrder = st.selectbox("DeliveryOrder", [0, 1])
    LoyaltyProgramMember = st.selectbox("LoyaltyProgramMember", [0, 1])
    WaitTime = st.number_input("WaitTime")
    ServiceRating = st.number_input("ServiceRating", step=1)
    FoodRating = st.number_input("FoodRating", step=1)
    AmbianceRating = st.number_input("AmbianceRating", step=1)
    
    submitted = st.form_submit_button("Predict")
    if submitted:
        payload = {
            "Age": Age,
            "Gender": Gender,
            "Income": Income,
            "VisitFrequency": VisitFrequency,
            "AverageSpend": AverageSpend,
            "PreferredCuisine": PreferredCuisine,
            "TimeOfVisit": TimeOfVisit,
            "GroupSize": GroupSize,
            "DiningOccasion": DiningOccasion,
            "MealType": MealType,
            "OnlineReservation": OnlineReservation,
            "DeliveryOrder": DeliveryOrder,
            "LoyaltyProgramMember": LoyaltyProgramMember,
            "WaitTime": WaitTime,
            "ServiceRating": ServiceRating,
            "FoodRating": FoodRating,
            "AmbianceRating": AmbianceRating
        }
        response = requests.post("http://localhost:8000/predict", json=payload)
        if response.status_code == 200:
            st.success("Prediction successful")
            st.write(response.json())
        else:
            st.error("Error in prediction")
            st.write(response.json())

# Section pour obtenir des informations sur le modèle
st.header("Model Info")
if st.button("Get Model Info"):
    response = requests.get("http://localhost:8000/model")
    if response.status_code == 200:
        st.write(response.json())
    else:
        st.error("Error getting model info")
        st.write(response.json())
