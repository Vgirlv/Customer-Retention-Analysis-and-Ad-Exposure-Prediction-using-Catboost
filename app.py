import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

st.set_page_config(page_title="Media Analytics Dashboard", layout="wide")

st.title("🎬 Media Analytics ML Dashboard")

@st.cache_resource
def load_models():
    churn_model = CatBoostClassifier()
    churn_model.load_model("churn_model.cbm")

    ad_model = CatBoostRegressor()
    ad_model.load_model("ad_model.cbm")

    return churn_model, ad_model

model_churn, model_reg = load_models()

option = st.sidebar.radio(
    "Select Module",
    ["Customer Churn Prediction", "Ad Exposure Prediction"]
)

# ==================================================
# CHURN PREDICTION
# ==================================================
if option == "Customer Churn Prediction":

    st.header("🔍 Customer Churn Prediction")

    watch_mean = st.number_input("Average Watch Time", 0.0)
    watch_sum = st.number_input("Total Watch Time", 0.0)
    ads_mean = st.number_input("Average Ads Seen", 0.0)
    ads_skipped_sum = st.number_input("Total Ads Skipped", 0.0)
    buffering_sum = st.number_input("Total Buffering Events", 0.0)
    rating_mean = st.slider("Average Rating", 0.0, 5.0, 3.0)

    if st.button("Predict Churn"):

        input_df = pd.DataFrame([[
            watch_mean,
            watch_sum,
            ads_mean,
            ads_skipped_sum,
            buffering_sum,
            rating_mean
        ]], columns=[
            "watch_mean",
            "watch_sum",
            "ads_mean",
            "ads_skipped_sum",
            "buffering_sum",
            "rating_mean"
        ])

        prob = model_churn.predict_proba(input_df)[0][1]

        st.subheader(f"Churn Probability: {round(prob, 3)}")

        if prob > 0.7:
            st.error("🚨 Critical Risk")
        elif prob > 0.5:
            st.warning("⚠️ High Risk")
        else:
            st.success("✅ Low Risk")

# ==================================================
# AD EXPOSURE PREDICTION
# ==================================================
elif option == "Ad Exposure Prediction":

    st.header("📢 Ad Exposure Prediction")

    subscription_type = st.selectbox("Subscription Type", ["Free", "Basic", "Premium"])
    watch_duration_minutes = st.number_input("Watch Duration (minutes)", 0.0)
    ad_skipped = st.number_input("Ads Skipped", 0.0)
    device_type = st.selectbox("Device Type", ["Mobile", "Tablet", "Desktop", "Smart TV"])
    network_type = st.selectbox("Network Type", ["WiFi", "4G", "5G"])
    buffering_events = st.number_input("Buffering Events", 0.0)

    if st.button("Predict Ad Exposure"):

        input_df = pd.DataFrame([[
            subscription_type,
            watch_duration_minutes,
            ad_skipped,
            device_type,
            network_type,
            buffering_events
        ]], columns=[
            "subscription_type",
            "watch_duration_minutes",
            "ad_skipped",
            "device_type",
            "network_type",
            "buffering_events"
        ])

        prediction = model_reg.predict(input_df)[0]

        st.subheader(f"Predicted Ad Exposure Count: {round(prediction, 2)}")
