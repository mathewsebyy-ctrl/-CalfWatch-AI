import streamlit as st

from risk_engine import calculate_risk


st.set_page_config(
    page_title="CalfWatch AI",
    page_icon="🐄",
    layout="wide"
)

st.title("🐄 CalfWatch AI")

st.subheader("Newborn Calf Health & Survival Risk Prediction")

st.info(
    "Early-warning decision-support prototype. "
    "Not a clinically validated veterinary diagnostic system."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    time_to_stand = st.number_input(
        "Time to Stand (seconds)",
        min_value=0,
        value=40
    )

with col2:
    time_to_suckle = st.number_input(
        "Time to First Suckle (seconds)",
        min_value=0,
        value=90
    )

if st.button("Calculate Risk"):

    risk, score = calculate_risk(
        time_to_stand,
        time_to_suckle
    )

    st.divider()

    st.metric(
        "Risk Score",
        score
    )

    if risk == "LOW":
        st.success("🟢 LOW RISK")

    elif risk == "MEDIUM":
        st.warning("🟡 MEDIUM RISK")

    else:
        st.error("🔴 HIGH RISK")
        st.warning(
            "Early-warning alert: delayed behaviour indicators detected."
        )