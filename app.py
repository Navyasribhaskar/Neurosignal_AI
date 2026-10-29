
import streamlit as st
import pandas as pd
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="NeuroSignal AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD RESULTS ----------------
RESULTS_FILE = Path(__file__).parent / "neurosignal_results.csv"


@st.cache_data
def load_results(file_path, modified_time):
    data = pd.read_csv(file_path)

    required = {"Model", "Metric", "Mean score"}
    if not required.issubset(data.columns):
        raise ValueError(
            "The results CSV is missing required columns: "
            + ", ".join(sorted(required))
        )

    data["Mean score"] = pd.to_numeric(
        data["Mean score"], errors="coerce"
    )
    data = data.dropna(subset=["Mean score"])

    if "Standard deviation" not in data.columns:
        data["Standard deviation"] = 0.0

    data["Standard deviation"] = pd.to_numeric(
        data["Standard deviation"], errors="coerce"
    ).fillna(0.0)

    return data


st.sidebar.title("🧠 NeuroSignal AI")
st.sidebar.caption("EEG Research & Analytics")
page = st.sidebar.radio(
    "Explore dashboard",
    [
        "Overview",
        "Model Explorer",
        "Results Table",
        "Methodology"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Research prototype • Not for medical use")

# ---------------- HEADER ----------------
st.title("🧠 NeuroSignal AI")
st.markdown(
    "### EEG Motor-Imagery Classification & Research Analytics"
)
st.write(
    "Explore classical machine-learning experiments on EEG "
    "recordings, compare model metrics, and inspect the "
    "results of participant-held-out evaluation."
)

if not RESULTS_FILE.exists():
    st.error(
        "Results file not found. Keep "
        "`neurosignal_results.csv` in the same folder as `app.py`."
    )
    st.stop()

try:
    df = load_results(
        str(RESULTS_FILE),
        RESULTS_FILE.stat().st_mtime
    )
except Exception as exc:
    st.error(f"Could not load the evaluation results: {exc}")
    st.stop()

models = sorted(df["Model"].unique())
metrics = sorted(df["Metric"].unique())

if not models or not metrics:
    st.error("No valid model results were found in the CSV.")
    st.stop()

# ---------------- OVERVIEW ----------------
if page == "Overview":
    st.subheader("Experiment at a glance")

    st.caption(
        "The following results are from the saved 10-participant "
        "experiment, not from a live EEG recording."
    )

    csp_name = "CSP + Logistic Regression"
    baseline_name = "Majority-class baseline"

    def get_score(model_name, metric_name):
        rows = df[
            (df["Model"] == model_name)
            & (df["Metric"] == metric_name)
        ]
        if rows.empty:
            return None
        return float(rows["Mean score"].iloc[0])

    accuracy = get_score(csp_name, "accuracy")
    balanced = get_score(csp_name, "balanced_accuracy")
    f1 = get_score(csp_name, "f1")
    baseline_accuracy = get_score(baseline_name, "accuracy")

    c1, c2, c3 = st.columns(3)

    if accuracy is not None:
        c1.metric("CSP Accuracy", f"{accuracy:.1%}")
    else:
        c1.metric("CSP Accuracy", "N/A")

    if balanced is not None:
        c2.metric("Balanced Accuracy", f"{balanced:.1%}")
    else:
        c2.metric("Balanced Accuracy", "N/A")

    if f1 is not None:
        c3.metric("F1 Score", f"{f1:.1%}")
    else:
        c3.metric("F1 Score", "N/A")

    st.subheader("Compare model performance")

    selected_metric = st.selectbox(
        "Choose an evaluation metric",
        metrics,
        index=metrics.index("accuracy")
        if "accuracy" in metrics else 0
    )

    chart_data = (
        df[df["Metric"] == selected_metric]
        .set_index("Model")[["Mean score"]]
    )

    st.bar_chart(chart_data, y="Mean score")

    st.caption(
        "Scores range from 0 to 1. Cross-validation standard "
        "deviations are available in the Results Table."
    )

    if accuracy is not None and baseline_accuracy is not None:
        st.markdown("#### Research interpretation")

        difference = accuracy - baseline_accuracy

        if difference > 0:
            st.info(
                f"For this metric, CSP accuracy is "
                f"{difference:.1%} higher than baseline accuracy. "
                "This alone does not establish generalization."
            )
        elif difference < 0:
            st.warning(
                f"CSP accuracy is {abs(difference):.1%} lower "
                "than the majority-class baseline. The current "
                "experiment does not demonstrate improvement "
                "over the baseline."
            )
        else:
            st.info("CSP and baseline accuracy are equal.")

    st.warning(
        "Research limitation: only 10 participants were used. "
        "The participant-held-out results are preliminary and "
        "do not establish reliable EEG classification. "
        "This is not a clinical diagnostic system."
    )

# ---------------- MODEL EXPLORER ----------------
elif page == "Model Explorer":
    st.subheader("Interactive Model Explorer")

    chosen_model = st.selectbox("Choose a model", models)
    model_data = df[df["Model"] == chosen_model].copy()

    for _, row in model_data.iterrows():
        score = float(row["Mean score"])
        st.write(f"**{row['Metric'].replace('_', ' ').title()}**")
        st.progress(max(0.0, min(1.0, score)))
        st.write(f"Score: {score:.3f} ({score:.1%})")
        st.write(
            "Cross-validation standard deviation: "
            f"{row['Standard deviation']:.3f}"
        )
        st.divider()

    st.caption(
        "A score is an evaluation result, not a confidence "
        "estimate for an individual prediction."
    )

# ---------------- RESULTS TABLE ----------------
elif page == "Results Table":
    st.subheader("Detailed Evaluation Results")

    chosen_metrics = st.multiselect(
        "Filter metrics",
        metrics,
        default=metrics
    )

    filtered = df[df["Metric"].isin(chosen_metrics)].copy()

    st.dataframe(
        filtered.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        label="⬇️ Download displayed results as CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="neurosignal_filtered_results.csv",
        mime="text/csv"
    )

    st.download_button(
        label="⬇️ Download complete results",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="neurosignal_results.csv",
        mime="text/csv"
    )

# ---------------- METHODOLOGY ----------------
elif page == "Methodology":
    st.subheader("How the experiment works")

    st.markdown("""
    **Dataset:** PhysioNet EEG Motor Movement/Imagery Dataset.

    **Subset:** Recordings from 10 participants, using one
    selected recording run per participant.

    **Preprocessing:** EEG signals filtered to 8–30 Hz for
    the CSP experiment. Annotated left- and right-fist
    motor-imagery trials were selected.

    **Feature extraction:** Common Spatial Patterns (CSP)
    extracts spatial features that distinguish the two classes.

    **Classifier:** Logistic Regression.

    **Evaluation:** Five-fold GroupKFold keeps each participant's
    trials together, so test participants are excluded from
    training in their respective fold.

    **Baseline:** A majority-class DummyClassifier provides a
    simple reference for interpreting model performance.
    """)

    st.markdown("#### Dataset source")
    st.link_button(
        "Open PhysioNet dataset",
        "https://physionet.org/content/eegmmidb/"
    )

    st.markdown("#### Limitations and responsible use")
    st.write(
        "This is a small research prototype. The current "
        "participant-held-out result does not outperform the "
        "majority-class baseline. More participants, recording "
        "runs, and validation are required before drawing "
        "broader conclusions. The app is not for medical use."
    )

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "NeuroSignal AI | Python • MNE-Python • scikit-learn • "
    "Streamlit | Educational and research purposes only"
)

