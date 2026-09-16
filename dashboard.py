"""
CalfWatch AI — Newborn calf monitoring dashboard.

Early-warning decision support built on computer-vision behaviour
indicators. Not a clinically validated veterinary diagnostic system.
"""

import os
import time
from datetime import datetime
from html import escape

import cv2
import streamlit as st

from risk_engine import calculate_risk
from calf_detection import detect_calf
import behaviour_detection


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CalfWatch AI",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONSTANTS
# ============================================================

VIDEO_PATH = "data/test_videos/calf_test.mp4.mp4"

# Render every Nth frame to the browser. Detection still runs on every
# frame; this only throttles the UI, which is the real bottleneck.
UI_FRAME_STRIDE = 3

VALID_RISK_LEVELS = {"LOW", "MEDIUM", "HIGH"}

RISK_PRESENTATION = {
    "HIGH": {
        "accent": "#B3261E",
        "tint": "rgba(179, 38, 30, 0.10)",
        "heading": "High risk",
        "summary": (
            "Behaviour timings fall outside the expected range for a "
            "healthy newborn."
        ),
        "action": (
            "Check the calf now. Confirm it is warm, responsive, and "
            "able to nurse. Escalate to a veterinarian if it cannot."
        ),
    },
    "MEDIUM": {
        "accent": "#8A5A00",
        "tint": "rgba(138, 90, 0, 0.10)",
        "heading": "Medium risk",
        "summary": (
            "One or more timings are delayed relative to the expected "
            "range."
        ),
        "action": (
            "Observe more frequently over the next few hours. Confirm "
            "the calf stands and suckles without assistance."
        ),
    },
    "LOW": {
        "accent": "#1B5E20",
        "tint": "rgba(27, 94, 32, 0.10)",
        "heading": "Low risk",
        "summary": "Behaviour timings fall within the expected range.",
        "action": "Continue routine observation.",
    },
}


# ============================================================
# STYLES
# ============================================================
# Colour is kept theme-neutral so the dashboard stays legible in both
# light and dark mode. Hardcoded white cards and grey text break
# entirely once a viewer switches to Streamlit's dark theme.

st.markdown(
    """
<style>

.block-container {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1440px;
}

.masthead {
    display: flex;
    align-items: baseline;
    gap: 14px;
    flex-wrap: wrap;
    margin-bottom: 4px;
}

.masthead-name {
    font-size: 34px;
    font-weight: 700;
    letter-spacing: -0.4px;
    line-height: 1.1;
}

.masthead-stage {
    font-size: 12px;
    font-weight: 600;
    padding: 3px 9px;
    border-radius: 999px;
    border: 1px solid rgba(128, 128, 128, 0.45);
    opacity: 0.8;
}

.masthead-line {
    font-size: 16px;
    opacity: 0.72;
    margin-bottom: 22px;
    max-width: 68ch;
}

.section-rule {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 34px 0 16px 0;
}

.section-rule h2 {
    font-size: 19px;
    font-weight: 600;
    margin: 0;
    white-space: nowrap;
}

.section-rule::after {
    content: "";
    flex: 1;
    height: 1px;
    background: rgba(128, 128, 128, 0.28);
}

.status-card {
    padding: 16px 18px;
    border-radius: 10px;
    border: 1px solid rgba(128, 128, 128, 0.28);
    border-left: 4px solid var(--card-accent, rgba(128, 128, 128, 0.5));
    min-height: 104px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 6px;
}

.status-title {
    font-size: 13px;
    opacity: 0.68;
    line-height: 1.3;
}

.status-value {
    font-size: 23px;
    font-weight: 600;
    line-height: 1.2;
    word-break: break-word;
}

.risk-panel {
    padding: 22px 24px;
    border-radius: 12px;
    border: 1px solid rgba(128, 128, 128, 0.28);
    border-left: 5px solid var(--risk-accent);
    background: var(--risk-tint);
}

.risk-panel-heading {
    font-size: 25px;
    font-weight: 700;
    color: var(--risk-accent);
    margin-bottom: 8px;
}

.risk-panel-summary {
    font-size: 15px;
    opacity: 0.88;
    margin-bottom: 14px;
    max-width: 72ch;
}

.risk-panel-action {
    font-size: 15px;
    padding-top: 12px;
    border-top: 1px solid rgba(128, 128, 128, 0.28);
    max-width: 72ch;
}

.risk-panel-action span {
    font-weight: 600;
}

.pending-panel {
    padding: 18px 20px;
    border-radius: 10px;
    border: 1px dashed rgba(128, 128, 128, 0.45);
    opacity: 0.85;
    max-width: 72ch;
}

.limitation-note {
    font-size: 13px;
    opacity: 0.62;
    margin-top: 10px;
    max-width: 78ch;
}

.footer {
    text-align: center;
    opacity: 0.55;
    font-size: 13px;
    margin-top: 30px;
    line-height: 1.7;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# HELPERS
# ============================================================

def safe(value):
    """Escape a value before interpolating it into raw HTML."""
    return escape(str(value))


def section(title):
    """Render a section heading followed by a rule."""
    st.markdown(
        f'<div class="section-rule"><h2>{safe(title)}</h2></div>',
        unsafe_allow_html=True
    )


def status_card(title, value, accent=None):
    """Render one status card with an optional accent colour."""
    accent_style = f"--card-accent:{accent};" if accent else ""

    st.markdown(
        f"""
        <div class="status-card" style="{accent_style}">
            <div class="status-title">{safe(title)}</div>
            <div class="status-value">{safe(value)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def format_seconds(value):
    """Format a seconds value, or report that it was never observed."""
    if value is None:
        return "Not observed"

    try:
        return f"{float(value):.1f} s"
    except (TypeError, ValueError):
        return "Not observed"


def derive_level(score):
    """Band a bare numeric score into a risk level."""
    try:
        numeric = float(score)
    except (TypeError, ValueError):
        return "UNKNOWN"

    # risk_engine uses a 0-100 scale; earlier prototypes used 0-6.
    if numeric <= 6:
        if numeric >= 4:
            return "HIGH"
        if numeric >= 2:
            return "MEDIUM"
        return "LOW"

    if numeric >= 55:
        return "HIGH"
    if numeric >= 25:
        return "MEDIUM"
    return "LOW"


def normalise_risk_result(result):
    """
    Return (level, score) from calculate_risk, whichever order it used.

    calculate_risk returns (risk, score). Reading by position silently
    swaps the two if that ever changes, so identify the level by its
    value and treat the remaining element as the score.
    """
    if isinstance(result, (list, tuple)):
        level = None
        score = None

        for item in result:
            if isinstance(item, str) and item.strip().upper() in VALID_RISK_LEVELS:
                level = item.strip().upper()
            elif isinstance(item, (int, float)) and not isinstance(item, bool):
                score = item

        if level is not None:
            return level, score

        score = result[0] if result else None
        return derive_level(score), score

    if isinstance(result, str):
        candidate = result.strip().upper()
        return (candidate, None) if candidate in VALID_RISK_LEVELS else ("UNKNOWN", None)

    return derive_level(result), result


def read_calf_flag(calf_result):
    """Interpret whatever detect_calf returned as a boolean."""
    if calf_result is None:
        return False

    if isinstance(calf_result, bool):
        return calf_result

    if isinstance(calf_result, dict):
        return bool(calf_result.get("calf_detected", False))

    if isinstance(calf_result, (list, tuple)):
        return len(calf_result) > 0

    return bool(calf_result)


def reset_state():
    """Clear all monitoring state back to its initial values."""
    for key, default in DEFAULT_STATE.items():
        st.session_state[key] = default

    for attribute, value in (
        ("time_to_stand", None),
        ("time_to_suckle", None),
        ("current_behavior", "Not started"),
    ):
        if hasattr(behaviour_detection, attribute):
            setattr(behaviour_detection, attribute, value)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "calf_detected": False,
    "current_behavior": "Not started",
    "time_to_stand": None,
    "time_to_suckle": None,
    "risk_score": None,
    "risk_level": "UNKNOWN",
    "monitoring_status": "Ready",
    "processed": False,
}

for state_key, state_default in DEFAULT_STATE.items():
    if state_key not in st.session_state:
        st.session_state[state_key] = state_default


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="masthead">
        <div class="masthead-name">CalfWatch AI</div>
        <div class="masthead-stage">Prototype</div>
    </div>
    <div class="masthead-line">
        Automated monitoring of newborn calf behaviour, scored for early
        signs of survival risk.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

video_exists = os.path.exists(VIDEO_PATH)

with st.sidebar:

    st.subheader("Monitoring mode")

    monitoring_mode = st.radio(
        "Mode",
        ["Test video", "Manual assessment"],
        label_visibility="collapsed"
    )

    st.divider()

    st.subheader("Video source")

    if video_exists:
        st.success("Test video found")
    else:
        st.error("Test video not found")

    st.caption(VIDEO_PATH)

    st.divider()

    st.subheader("Pipeline")

    st.caption("Detection — YOLO-based calf detection")
    st.caption("Behaviour — vision-based classification")
    st.caption("Scoring — rule-based risk engine")

    st.divider()

    if st.button("Reset monitoring", use_container_width=True):
        reset_state()
        st.rerun()

    st.caption("SIH 2026 prototype")


# ============================================================
# MANUAL ASSESSMENT MODE
# ============================================================

if monitoring_mode == "Manual assessment":

    section("Manual assessment")

    st.write(
        "Enter observed timings to score a calf without running video "
        "monitoring."
    )

    input_col1, input_col2 = st.columns(2)

    with input_col1:
        stand_time = st.number_input(
            "Time to stand (seconds)",
            min_value=0.0,
            value=40.0,
            step=1.0
        )

    with input_col2:
        suckle_time = st.number_input(
            "Time to first suckle (seconds)",
            min_value=0.0,
            value=90.0,
            step=1.0
        )

    if st.button("Calculate risk", type="primary"):

        try:
            level, score = normalise_risk_result(
                calculate_risk(stand_time, suckle_time)
            )

            st.session_state.time_to_stand = stand_time
            st.session_state.time_to_suckle = suckle_time
            st.session_state.risk_level = level
            st.session_state.risk_score = score
            st.session_state.current_behavior = "Manually entered"
            st.session_state.monitoring_status = "Completed"
            st.session_state.processed = True

        except Exception as error:
            st.error(
                f"Risk could not be calculated: {error}. Check that "
                "risk_engine.calculate_risk accepts two numeric arguments."
            )


# ============================================================
# VIDEO MONITORING MODE
# ============================================================

else:

    section("Video monitoring")

    if not video_exists:

        st.error(
            "No test video at the expected path. Place a video file "
            "there, or switch to manual assessment in the sidebar."
        )
        st.code(VIDEO_PATH)

    else:

        video_col, control_col = st.columns([2.6, 1])

        with control_col:
            start_monitoring = st.button(
                "Start monitoring",
                type="primary",
                use_container_width=True
            )
            st.caption(
                "Processes the full video, tracking when the calf first "
                "stands and first suckles."
            )

        with video_col:
            video_placeholder = st.empty()

        if start_monitoring:

            st.session_state.monitoring_status = "Monitoring"

            capture = cv2.VideoCapture(VIDEO_PATH)

            if not capture.isOpened():
                st.error(
                    "The video file could not be opened. It may be "
                    "corrupt or in an unsupported codec."
                )

            else:
                fps = capture.get(cv2.CAP_PROP_FPS)
                if not fps or fps <= 0:
                    fps = 30.0

                total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

                frame_number = 0
                progress = st.progress(0.0)
                progress_label = st.empty()

                try:
                    while True:
                        ret, frame = capture.read()
                        if not ret:
                            break

                        frame_number += 1
                        elapsed_time = frame_number / fps

                        # --- Calf detection ---------------------
                        try:
                            calf_result = detect_calf(frame)
                        except Exception as error:
                            calf_result = None
                            progress_label.warning(
                                f"Detection failed on frame "
                                f"{frame_number}: {error}"
                            )

                        calf_found = read_calf_flag(calf_result)
                        st.session_state.calf_detected = calf_found

                        # --- Behaviour classification -----------
                        try:
                            behaviour_result = (
                                behaviour_detection.detect_behavior(
                                    frame, calf_result, elapsed_time
                                )
                            )
                            if isinstance(behaviour_result, str):
                                st.session_state.current_behavior = (
                                    behaviour_result
                                )
                        except Exception:
                            pass

                        # --- Timings from behaviour module ------
                        for attribute in ("time_to_stand", "time_to_suckle"):
                            if hasattr(behaviour_detection, attribute):
                                st.session_state[attribute] = getattr(
                                    behaviour_detection, attribute
                                )

                        # --- Render (throttled) -----------------
                        if frame_number % UI_FRAME_STRIDE == 0:
                            display = frame.copy()

                            if calf_found:
                                banner = "CALF DETECTED"
                                colour = (60, 170, 60)
                            else:
                                banner = " DETECTED"
                                colour = (60, 60, 210)

                            cv2.putText(
                                display, banner, (15, 34),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.85, colour, 2
                            )
                            cv2.putText(
                                display,
                                f"Behaviour: "
                                f"{st.session_state.current_behavior}",
                                (15, 68),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (245, 245, 245), 2
                            )
                            cv2.putText(
                                display,
                                f"Elapsed: {elapsed_time:.1f}s",
                                (15, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (245, 245, 245), 2
                            )

                            video_placeholder.image(
                                cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
                                channels="RGB",
                                use_container_width=True
                            )

                            if total_frames > 0:
                                progress.progress(
                                    min(frame_number / total_frames, 1.0)
                                )
                                progress_label.caption(
                                    f"Frame {frame_number} of "
                                    f"{total_frames} — "
                                    f"{elapsed_time:.1f}s elapsed"
                                )
                            else:
                                progress_label.caption(
                                    f"Frame {frame_number} — "
                                    f"{elapsed_time:.1f}s elapsed"
                                )

                            time.sleep(0.01)

                finally:
                    capture.release()

                progress.progress(1.0)
                progress_label.empty()
                st.session_state.monitoring_status = "Completed"

                # --- Final scoring --------------------------
                stand_value = st.session_state.time_to_stand
                suckle_value = st.session_state.time_to_suckle

                if stand_value is not None and suckle_value is not None:
                    try:
                        level, score = normalise_risk_result(
                            calculate_risk(stand_value, suckle_value)
                        )
                        st.session_state.risk_level = level
                        st.session_state.risk_score = score
                    except Exception as error:
                        st.warning(
                            f"Risk could not be calculated: {error}"
                        )
                else:
                    st.warning(
                        "Monitoring finished, but standing or suckling "
                        "was never observed, so no risk score was "
                        "produced."
                    )

                st.session_state.processed = True


# ============================================================
# MONITORING OVERVIEW
# ============================================================

section("Monitoring overview")

risk_level = str(st.session_state.risk_level).upper()
risk_style = RISK_PRESENTATION.get(risk_level)

overview_cols = st.columns(4)

with overview_cols[0]:
    detected = st.session_state.calf_detected
    status_card(
        "Calf detection",
        "Detected" if detected else "Not detected",
        accent="#1B5E20" if detected else None
    )

with overview_cols[1]:
    status_card("Current behaviour", st.session_state.current_behavior)

with overview_cols[2]:
    status_card("Monitoring status", st.session_state.monitoring_status)

with overview_cols[3]:
    status_card(
        "Risk level",
        risk_style["heading"] if risk_style else "Not scored",
        accent=risk_style["accent"] if risk_style else None
    )


# ============================================================
# TIMINGS
# ============================================================

section("Behaviour timings")

timing_cols = st.columns(3)

with timing_cols[0]:
    st.metric(
        "Time to stand",
        format_seconds(st.session_state.time_to_stand)
    )

with timing_cols[1]:
    st.metric(
        "Time to first suckle",
        format_seconds(st.session_state.time_to_suckle)
    )

with timing_cols[2]:
    risk_score = st.session_state.risk_score
    st.metric(
        "Risk score",
        "Not scored" if risk_score is None else str(risk_score)
    )


# ============================================================
# RISK ASSESSMENT
# ============================================================

section("Risk assessment")

if risk_style:
    st.markdown(
        f"""
        <div class="risk-panel"
             style="--risk-accent:{risk_style['accent']};
                    --risk-tint:{risk_style['tint']};">
            <div class="risk-panel-heading">
                {safe(risk_style['heading'])}
            </div>
            <div class="risk-panel-summary">
                {safe(risk_style['summary'])}
            </div>
            <div class="risk-panel-action">
                <span>Recommended action.</span>
                {safe(risk_style['action'])}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="limitation-note">'
        'Risk thresholds are prototype rules derived from behaviour '
        'timing alone. They have not been clinically validated and do '
        'not replace veterinary assessment.'
        '</div>',
        unsafe_allow_html=True
    )

else:
    st.markdown(
        """
        <div class="pending-panel">
            No risk score yet. Run video monitoring, or switch to manual
            assessment in the sidebar and enter observed timings.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    f"""
    <div class="footer">
        Last updated {datetime.now().strftime('%d %b %Y, %H:%M:%S')}<br>
        CalfWatch AI — SIH 2026 prototype. Early-warning decision
        support, not a veterinary diagnostic system.
    </div>
    """,
    unsafe_allow_html=True
)