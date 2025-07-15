import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("smart_mobility_dataset.csv")

# Extract hour from Timestamp
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df["Hour"] = df["Timestamp"].dt.hour

# Encode categorical features
le_weather = LabelEncoder()
le_traffic = LabelEncoder()
df["Weather_Condition_Code"] = le_weather.fit_transform(df["Weather_Condition"])
df["Traffic_Condition_Code"] = le_traffic.fit_transform(df["Traffic_Condition"])

# Features and targets
features = ["Hour", "Latitude", "Longitude", "Weather_Condition_Code"]
X = df[features]

y_class = df["Traffic_Condition_Code"]
y_reg = df["Vehicle_Count"]

# Split and train models
X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
_, _, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_class_train)

reg = RandomForestRegressor()
reg.fit(X_train, y_reg_train)

# App UI
st.title("🚦 AI-Based Traffic Congestion Predictor")

# Sidebar Inputs
st.sidebar.header("Enter Input Features")
hour = st.sidebar.slider("Hour of the Day", 0, 23, 8)
lat = st.sidebar.number_input("Latitude", value=40.75)
lon = st.sidebar.number_input("Longitude", value=-73.95)
weather = st.sidebar.selectbox("Weather Condition", le_weather.classes_)

# Prepare input for prediction
weather_code = le_weather.transform([weather])[0]
input_data = np.array([[hour, lat, lon, weather_code]])

# Predictions
if st.button("Predict Traffic"):
    traffic_pred_code = clf.predict(input_data)[0]
    traffic_pred = le_traffic.inverse_transform([traffic_pred_code])[0]
    vehicle_pred = reg.predict(input_data)[0]

    st.success(f"🚗 Predicted Traffic Condition: **{traffic_pred}**")
    st.success(f"📊 Predicted Vehicle Count: **{int(vehicle_pred)}** vehicles")

# Visualizations
st.subheader("📈 Traffic Volume vs Time of Day")
fig, ax = plt.subplots()
sns.boxplot(x="Hour", y="Vehicle_Count", data=df, ax=ax)
st.pyplot(fig)

st.subheader("🌦️ Traffic Condition Distribution by Weather")
fig2, ax2 = plt.subplots()
sns.countplot(x="Weather_Condition", hue="Traffic_Condition", data=df, ax=ax2)
plt.xticks(rotation=45)
st.pyplot(fig2)

# Raw Data Option
if st.checkbox("Show Raw Dataset"):
    st.dataframe(df.head())
