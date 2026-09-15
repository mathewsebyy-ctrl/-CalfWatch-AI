import streamlit as st
import os
from pipeline import analyze_video

st.set_page_config(
    page_title="CalfWatch AI",
    page_icon="🐄",
    layout="wide"
)

# Note: You have ".mp4.mp4" in the path. You may want to check if it should just be ".mp4"
VIDEO_PATH = "data/test_videos/calf_test.mp4.mp4"

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("🐄 CalfWatch AI")
st.subheader("Newborn Calf Health & Survival Risk Prediction")
st.info(
    "Early-warning decision-support prototype. "
    "Not a clinically validated veterinary diagnostic system."
)
st.divider()

# --------------------------------------------------
# VIDEO
# --------------------------------------------------
st.header("🎥 Calf Video Analysis")
video_col, control_col = st.columns([1.5, 1])

with video_col:
    if os.path.exists(VIDEO_PATH):
        st.video(VIDEO_PATH)
    else:
        st.error("Test video not found.")

with control_col:
    st.subheader("🤖 AI Analysis")

    if st.button("▶ Run AI Analysis", type="primary"):
        with st.spinner("Analyzing calf video..."):
            result = analyze_video(VIDEO_PATH)

        if result is None:
            st.error("Could not open the video.")
        else:
            st.session_state["result"] = result
            st.success("Analysis completed successfully.")

st.divider()

# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------
if "result" not in st.session_state:
    st.info("Click 'Run AI Analysis' to analyze the calf video.")
else:
    status = st.session_state["result"]

    # --------------------------------------------------
    # CALF MONITORING
    # --------------------------------------------------
    st.header("🐄 Calf Monitoring")
    col1, col2, col3 = st.columns(3)

    with col1:
        if status["calf_detected"]:
            st.metric(
                "Calf Detection",
                "Detected"
            )
        else:
            st.metric(
                "Calf Detection",
                "Not Detected"
            )

    with col2:
        st.metric(
            "Current Behaviour",
            status["current_behavior"]
        )

    with col3:
        st.metric(
            "Monitoring Status",
            status["monitoring_status"]
        )

    st.divider()

    # --------------------------------------------------
    # BEHAVIOUR TIMING
    # --------------------------------------------------
    st.header("⏱️ Behaviour Timing")
    time_col1, time_col2 = st.columns(2)

    with time_col1:
        if status["time_to_stand"] is not None:
            st.metric(
                "Time to Stand",
                f'{status["time_to_stand"]:.2f} sec'
            )
        else:
            st.metric(
                "Time to Stand",
                "Not detected"
            )

    with time_col2:
        if status["time_to_suckle"] is not None:
            st.metric(
                "Time to First Suckle",
                f'{status["time_to_suckle"]:.2f} sec'
            )
        else:
            st.metric(
                "Time to First Suckle",
                "Not detected"
            )

    st.divider()

    # --------------------------------------------------
    # RISK ANALYSIS
    # --------------------------------------------------
    st.header("⚠️ Health Risk Analysis")
    risk_col1, risk_col2 = st.columns(2)

    with risk_col1:
        st.metric(
            "Risk Score",
            status["risk_score"]
        )

    with risk_col2:
        st.metric(
            "Risk Level",
            status["risk_level"]
        )

    risk = status["risk_level"]

    if risk == "LOW":
        st.success(
            "🟢 LOW RISK — Calf behaviour indicators "
            "are currently within the prototype's expected range."
        )
    elif risk == "MEDIUM":
        st.warning(
            "🟡 MEDIUM RISK — Monitor the calf closely "
            "for delayed behaviour indicators."
        )
    else:
        st.error(
            "🔴 HIGH RISK — Immediate attention recommended."
        )
        st.warning(
            "Early-warning alert: delayed behaviour "
            "indicators detected."
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
