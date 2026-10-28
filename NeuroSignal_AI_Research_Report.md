# NeuroSignal AI
## EEG Classification and Research Analytics

### 1. Project Overview
NeuroSignal AI is an experimental machine-learning
project investigating EEG motor-imagery classification
and generalization to unseen participants.

### 2. Objective
The objective is to build a reproducible EEG pipeline,
compare classical ML methods, and investigate whether
a classifier generalizes to participants excluded
from training.

### 3. Dataset
Source: PhysioNet EEG Motor Movement/Imagery Dataset
https://physionet.org/content/eegmmidb/

The initial experiment used recordings from 10 participants,
with 15 selected trials per participant, giving 150 trials.

### 4. Methodology
1. Load EEG recordings using MNE-Python.
2. Select annotated motor-imagery trials.
3. Apply an 8–30 Hz band-pass filter.
4. Extract Common Spatial Patterns (CSP).
5. Train Logistic Regression.
6. Evaluate using five-fold participant-held-out GroupKFold.
7. Compare results with a majority-class baseline.

### 5. Results
CSP + Logistic Regression:
- Accuracy: 50.7%
- Balanced accuracy: 50.6%
- F1 score: 34.5%

Majority-class baseline:
- Accuracy: 51.3%
- Balanced accuracy: 50.0%
- F1 score: 0.0%

These are preliminary results from a small dataset subset.
The model did not demonstrate useful generalization
beyond the majority-class baseline.

### 6. Interpretation
The experiment demonstrates an end-to-end EEG processing
and evaluation workflow. The results also illustrate why
participant-independent testing is important.

The current findings do not establish reliable EEG
classification performance.

### 7. Limitations
- Only 10 participants were included.
- Only one recording run per participant was used.
- Results may vary with different participants and runs.
- The evaluation is preliminary and requires further study.
- The application is not intended for clinical use.

### 8. Future Work
- Include more participants and recording runs.
- Compare additional EEG preprocessing methods.
- Investigate feature selection and model calibration.
- Improve automated evaluation and reproducibility.
- Test the application in a clean environment.

### 9. Technology Stack
Python, MNE-Python, NumPy, Pandas, SciPy,
scikit-learn, Matplotlib and Streamlit.

### 10. Conclusion
NeuroSignal AI provides a working experimental framework
for EEG preprocessing, classical ML classification,
evaluation and visualization. Its current results motivate
further research into generalization across participants.

---
Educational and research purposes only.
Not a medical diagnostic system.
