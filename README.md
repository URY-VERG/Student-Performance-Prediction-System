# Student Performance Prediction System — Advanced Version

## Features
- Premium Student Performance Intelligence dashboard with saved-prediction KPIs, filters, model metrics, activity, and clearly labeled synthetic training data
- Individual student prediction and local SQLite record storage
- Search, filter, paginate, view/edit, duplicate-ID checks, confirmed delete, and CSV export of saved student records
- Batch CSV preview, validation, missing-value review, predictions, summary statistics, and downloadable results
- Three ML regressors: Linear Regression, Decision Tree, Random Forest
- Model comparison using MAE, RMSE, R²
- College Result Analysis for Excel/CSV, supported two-header workbook parsing, column mapping, summaries, subject analysis, PDF/CSV exports
- Configurable support-risk review using available saved predictions
- Reports center for student, batch, college, and model exports
- Downloadable per-student PDF report and linear-model feature contribution explanation
- Settings, methodology, responsible-use notes, and dataset preview

The application loads the existing `student_models.joblib`, `model_metrics.json`, and `students.db`; it does not retrain models automatically on app startup. To intentionally retrain, run `python train_model.py`.

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
8. Keep the existing model files and database; the app does not retrain on launch. Only if you intentionally want to retrain, run `python train_model.py` first.
9. Run the app:
   ```powershell
   python -m streamlit run app.py
   ```
10. Open `http://localhost:8501`.

Run the app from the project folder with:

```powershell
python -m streamlit run app.py
```

If missing, the app creates the `students.db` table without deleting existing records. The training script creates `student_models.joblib` and updates `model_metrics.json` when intentionally run.

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
