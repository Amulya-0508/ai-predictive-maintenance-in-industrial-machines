@echo off
python src\generate_data.py
python src\train_model.py
streamlit run dashboard\app.py
