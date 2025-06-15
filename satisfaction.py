import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Title
st.title("Passenger Satisfaction Prediction App")

# --- Load Artifacts ---
@st.cache_data(show_spinner=True)
def load_artifacts():
    base_path = r"C:\py project\venv\Passenger Satisfaction"
    with open(os.path.join(base_path, "final_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(base_path, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(base_path, "label_encoders.pkl"), "rb") as f:
        label_encoders = pickle.load(f)
    with open(os.path.join(base_path, "expected_columns.pkl"), "rb") as f:
        expected_columns = pickle.load(f)
    return model, scaler, label_encoders, expected_columns

@st.cache_data
def load_raw_data():
    csv_path = r"C:\py project\venv\Passenger Satisfaction\Passenger_Satisfaction.csv"
    return pd.read_csv(csv_path)

# Load everything
model, scaler, label_encoders, expected_columns = load_artifacts()
df_raw = load_raw_data()

# --- Visualization Section ---
st.header("Customer Satisfaction Trends")

# Satisfaction distribution
st.subheader("Satisfaction Distribution")
fig, ax = plt.subplots()
sns.countplot(data=df_raw, x="satisfaction", palette="Set2", ax=ax)
ax.set_title("Satisfaction Distribution")
st.pyplot(fig)

# Satisfaction by Gender
st.subheader("Satisfaction by Gender")
fig2, ax2 = plt.subplots()
sns.countplot(data=df_raw, x="Gender", hue="satisfaction", palette="Set1", ax=ax2)
ax2.set_title("Satisfaction by Gender")
st.pyplot(fig2)

# --- Input Section ---
st.header("Input Customer Features")

def user_input_features():
    gender = st.selectbox("Gender", options=df_raw["Gender"].unique())
    customer_type = st.selectbox("Customer Type", options=df_raw["Customer Type"].unique())
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    type_of_travel = st.selectbox("Type of Travel", options=df_raw["Type of Travel"].unique())
    class_type = st.selectbox("Class", options=df_raw["Class"].unique())
    
    inflight_wifi_service = st.slider("Inflight Wifi Service Rating (1-5)", 1, 5, 3)
    departure_arrival_time_convenient = st.slider("Departure/Arrival Time Convenience (1-5)", 1, 5, 3)
    ease_of_online_booking = st.slider("Ease of Online Booking (1-5)", 1, 5, 3)
    gate_location = st.slider("Gate Location Rating (1-5)", 1, 5, 3)
    food_and_drink = st.slider("Food and Drink Rating (1-5)", 1, 5, 3)
    online_boarding = st.slider("Online Boarding Rating (1-5)", 1, 5, 3)
    seat_comfort = st.slider("Seat Comfort Rating (1-5)", 1, 5, 3)
    inflight_entertainment = st.slider("Inflight Entertainment Rating (1-5)", 1, 5, 3)
    onboard_service = st.slider("Onboard Service Rating (1-5)", 1, 5, 3)
    leg_room = st.slider("Leg Room Rating (1-5)", 1, 5, 3)
    baggage_handling = st.slider("Baggage Handling Rating (1-5)", 1, 5, 3)
    checkin_service = st.slider("Check-in Service Rating (1-5)", 1, 5, 3)
    inflight_service = st.slider("Inflight Service Rating (1-5)", 1, 5, 3)
    cleanliness = st.slider("Cleanliness Rating (1-5)", 1, 5, 3)
    departure_delay = st.number_input("Departure Delay in Minutes", min_value=0, max_value=500, value=0)

    data = {
        "Gender": gender,
        "Customer Type": customer_type,
        "Age": age,
        "Type of Travel": type_of_travel,
        "Class": class_type,
        "Inflight wifi service": inflight_wifi_service,
        "Departure/Arrival time convenient": departure_arrival_time_convenient,
        "Ease of Online booking": ease_of_online_booking,
        "Gate location": gate_location,
        "Food and drink": food_and_drink,
        "Online boarding": online_boarding,
        "Seat comfort": seat_comfort,
        "Inflight entertainment": inflight_entertainment,
        "On-board service": onboard_service,
        "Leg room service": leg_room,
        "Baggage handling": baggage_handling,
        "Checkin service": checkin_service,
        "Inflight service": inflight_service,
        "Cleanliness": cleanliness,
        "Departure Delay in Minutes": departure_delay,
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# --- Preprocessing function ---
def preprocess_input(df_input, scaler, label_encoders, expected_columns):
    # Label encode categorical columns
    categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
    for col in categorical_cols:
        df_input[col] = label_encoders[col].transform(df_input[col])

    # Add missing columns if any (set to 0)
    for col in expected_columns:
        if col not in df_input.columns:
            df_input[col] = 0

    # Reorder columns as expected by model
    df_input = df_input[expected_columns]

    # Separate numeric columns (those NOT in categorical_cols)
    numeric_cols = [col for col in expected_columns if col not in categorical_cols]

    # Scale numeric columns only
    df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])

    return df_input.values  # return numpy array for prediction


# --- Prediction ---
if st.button("Predict Satisfaction"):
    try:
        processed_input = preprocess_input(input_df.copy(), scaler, label_encoders, expected_columns)
        prediction = model.predict(processed_input)
        proba = model.predict_proba(processed_input)[0][1]
        satisfaction_label = "Satisfied" if prediction[0] == 1 else "Neutral or Dissatisfied"
        st.write(f"### Prediction: {satisfaction_label}")
        st.write(f"Prediction Probability of Satisfaction: {proba:.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# --- Debug info ---
st.write("📁 Current working directory:", os.getcwd())
st.write("📂 Files in current directory:", os.listdir())
