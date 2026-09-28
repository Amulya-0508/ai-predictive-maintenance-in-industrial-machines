# AI-Based Predictive Maintenance for Industrial Machines

Ready-to-run software prototype using simulated industrial sensor telemetry and a Random Forest classifier.

## Features
- Simulated temperature, vibration, current and RPM sensors
- Automatic synthetic dataset generation
- Random Forest machine-condition classification
- Accuracy, precision, recall, F1-score and confusion matrix
- Live virtual sensor simulation
- Machine health score
- Failure-risk estimate
- Maintenance recommendations
- Streamlit dashboard
- Multiple virtual machines

## Important
This is a software prototype using simulated sensor readings. The readings are not from physical industrial sensors. The architecture can later accept ESP32, PLC, or industrial sensor data.

## Windows
Open Command Prompt in this folder and run:

    python -m venv .venv
    .venv\\Scripts\\activate
    pip install -r requirements.txt
    python src\\generate_data.py
    python src\\train_model.py
    streamlit run dashboard\\app.py

If `python` is not recognized, try `py`.
