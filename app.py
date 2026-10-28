import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="NeuroSignal AI",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 NeuroSignal AI")
st.subheader("EEG Classification & Research Analytics")

st.write(
    "An experimental EEG machine-learning project "
    "evaluating motor-imagery classification and "
    "generalization to unseen participants."
)

results_path = Path("neurosignal_results.csv")

if not results_path.exists():
    st.error(
        "Results file not found. Run the evaluation "
        "and CSV export cells in your Colab notebook first."
    )
    st.stop()

df = pd.read_csv(results_path)

# Select the results for the participant-held-out experiment
accuracy_row = df[
    (df["Model"] == "CSP + Logistic Regression") &
    (df["Metric"] == "accuracy")
]

balanced_row = df[
    (df["Model"] == "CSP + Logistic Regression") &
    (df["Metric"] == "balanced_accuracy")
]

f1_row = df[
    (df["Model"] == "CSP + Logistic Regression") &
    (df["Metric"] == "f1")
]

if accuracy_row.empty or balanced_row.empty or f1_row.empty:
    st.error("Expected evaluation metrics are missing from the CSV.")
    st.stop()

accuracy = accuracy_row["Mean score"].iloc[0]
balanced_accuracy = balanced_row["Mean score"].iloc[0]
f1 = f1_row["Mean score"].iloc[0]

st.markdown("### Model Performance")

c1, c2, c3 = st.columns(3)

c1.metric("Accuracy", f"{accuracy:.1%}")
c2.metric("Balanced Accuracy", f"{balanced_accuracy:.1%}")
c3.metric("F1 Score", f"{f1:.1%}")

st.caption(
    "These are preliminary participant-held-out results "
    "from 10 participants. They are not clinical performance "
    "estimates or evidence of a validated EEG classifier."
)

st.markdown("### Model Comparison")

pivot = df.pivot(
    index="Metric",
    columns="Model",
    values="Mean score"
)

st.bar_chart(pivot)

st.markdown("### Detailed Results")
st.dataframe(df.round(3), use_container_width=True)

st.markdown("### About this project")
st.write(
    "NeuroSignal AI uses MNE-Python for EEG processing, "
    "Common Spatial Patterns for feature extraction, and "
    "scikit-learn for classification and evaluation."
)

st.info(
    "Research limitation: performance on unseen participants "
    "is close to the majority-class baseline in this experiment. "
    "Further validation and model improvement are required."
)





st.markdown("### Research Findings")

st.write(
    "This experiment investigates whether an EEG classifier "
    "generalizes to participants excluded from training."
)

st.write(
    "The initial participant-held-out evaluation achieved "
    "50.7% accuracy, compared with 51.3% for the "
    "majority-class baseline."
)

st.warning(
    "The current experiment does not demonstrate useful "
    "generalization beyond the baseline. The dataset subset "
    "is small, and further validation is required."
)

st.markdown("### Experimental Setup")

st.write("""
- Dataset: PhysioNet EEG Motor Movement/Imagery
- Participants: 10
- Trials: 150
- Signal filtering: 8–30 Hz
- Feature extraction: Common Spatial Patterns (CSP)
- Classifier: Logistic Regression
- Evaluation: 5-fold participant-held-out GroupKFold
""")

