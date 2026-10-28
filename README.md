# NeuroSignal AI

**EEG Classification and Research Analytics**

NeuroSignal AI is an experimental machine-learning project
that investigates EEG motor-imagery classification and
generalization to unseen participants.

## Features
- EEG preprocessing with MNE-Python
- Frequency-band power features
- Common Spatial Patterns (CSP)
- Logistic Regression classification
- Participant-held-out evaluation
- Interactive Streamlit dashboard

## Technology
Python, MNE-Python, NumPy, Pandas, SciPy,
scikit-learn, Matplotlib and Streamlit.

## Dataset
PhysioNet EEG Motor Movement/Imagery Dataset:
https://physionet.org/content/eegmmidb/

## Preliminary results
The experiment used recordings from 10 participants.

- CSP + Logistic Regression accuracy: 50.7%
- Balanced accuracy: 50.6%
- F1 score: 34.5%
- Majority-class baseline accuracy: 51.3%

These results do not demonstrate useful generalization
beyond the baseline. Further evaluation is required.

## Run locally
Install dependencies:

    pip install -r requirements.txt

Start the dashboard:

    streamlit run app.py

Keep neurosignal_results.csv in the project directory.

## Limitations
This is a small experimental study, not a validated
EEG classifier or a medical diagnostic system.

## Future work
- Evaluate more participants and recording sessions
- Investigate signal preprocessing and feature selection
- Improve model evaluation and reproducibility

## Author
Navyasree

For educational and research purposes only.
