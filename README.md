# Student Performance Prediction System — Advanced Version

## Features
- Dashboard with dataset KPIs and charts
- Individual student prediction and local SQLite record storage
- Search, view, delete and CSV export of saved student records
- Batch predictions from an uploaded CSV and downloadable results
- Three ML regressors: Linear Regression, Decision Tree, Random Forest
- Model comparison using MAE, RMSE, R²
- Downloadable per-student PDF report
- At-risk/support flag based on configurable demonstration thresholds
- Methodology, limitations and dataset preview

## Dataset caveat
The included `student_performance.csv` is synthetic demonstration data, not real student data. Replace it with a suitable public or approved dataset before making real-world claims. Predictions are estimates, not a substitute for educator judgement.

## Update your existing project
1. Stop the running Streamlit app in the terminal using `Ctrl+C`.
2. Keep a backup copy of your existing project folder.
3. Extract this ZIP to a temporary folder.
4. Copy the new `app.py`, `train_model.py`, `requirements.txt`, `student_performance.csv`, and `PROJECT_REPORT.md` into your existing `student_performance_prediction` folder and choose **Replace** when asked.
5. Keep your existing `.venv` folder. Do not copy `.venv` from elsewhere.
6. Open VS Code terminal in the project folder and activate the environment if needed:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
7. Install/update packages:
   ```powershell
   python -m pip install -r requirements.txt
   ```
8. Retrain all models:
   ```powershell
   python train_model.py
   ```
9. Run the app:
   ```powershell
   python -m streamlit run app.py
   ```
10. Open `http://localhost:8501`.

The app creates `students.db` automatically for locally saved predictions. The training script creates `student_models.joblib` and updates `model_metrics.json`.

## Batch CSV format
Required columns:
- `study_hours_per_day`
- `attendance_percent`
- `previous_score`
- `assignment_average`
- `sleep_hours`
- `class_participation`

Optional extra columns (e.g. student name/ID) are preserved in batch output.

## Notes
- This is a classroom prototype, not a production student information system.
- Local database records are stored unencrypted; do not enter sensitive real student data.
- Do not interpret the demo score bands as official pass/fail decisions.
