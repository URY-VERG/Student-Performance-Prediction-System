# Student Performance Prediction System Using Machine Learning

## 1. Problem Definition
This project demonstrates a machine-learning application that estimates a student's final score from study hours, attendance, previous score, assignment average, sleep hours and class participation. It is intended for educational demonstration, not high-stakes decisions.

## 2. Objectives
- Inspect and preprocess a tabular dataset.
- Train and compare Linear Regression, Decision Tree Regression and Random Forest Regression.
- Evaluate models using MAE, RMSE and R².
- Build a Streamlit dashboard for individual and batch predictions.
- Save student prediction records locally, export CSV data and generate PDF reports.
- Present result analysis, limitations and future scope.

## 3. Dataset Collection
The included `student_performance.csv` has 500 synthetic sample rows generated for demonstrating the workflow. It is not real student data. For a research-quality report, replace it with a documented public dataset or institution-approved data and describe its source accurately.

## 4. Data Preprocessing
The application separates six input features from the target final score and uses an 80/20 train-test split with a fixed random state. Median imputation is included in model pipelines. StandardScaler is applied to Linear Regression. CSV uploads are checked for required columns and feature values are converted to numeric form.

## 5. Model Development
Three regression models are compared:
- **Linear Regression:** baseline linear relationship between inputs and score.
- **Decision Tree Regressor:** models non-linear decision rules.
- **Random Forest Regressor:** averages multiple decision trees to improve robustness.

## 6. Performance Evaluation
The training script prints and saves the following test-set metrics:
- **MAE:** average absolute prediction error; lower is better.
- **RMSE:** square root of mean squared error; lower is better and larger errors are penalized more.
- **R²:** goodness-of-fit measure; closer to 1 is generally better.

Insert the actual metrics printed by `python train_model.py` into the submitted report. Do not fabricate results. Because the dataset is synthetic, metrics demonstrate code execution, not real-world validity.

## 7. Model Optimization
The app selects the model with the highest R² on the fixed held-out split for this prototype. Baseline hyperparameters are fixed in the training script. Future work should use cross-validation and GridSearchCV/RandomizedSearchCV on training data, with final evaluation on an untouched test set.

## 8. Application Features
- Dashboard with dataset counts and score charts.
- Individual prediction with a performance band and supportive study recommendations.
- Local SQLite database for saved prediction records.
- Search, delete and CSV export of saved records.
- Batch prediction from a CSV file with downloadable results.
- PDF report download for individual predictions.
- Model comparison and methodology page.

## 9. Result Analysis
The application presents model metrics, a score distribution chart and a study-hours scatter plot. Prediction bands are demonstration labels only and should not be treated as official pass/fail outcomes. Any low-score or attendance flag should prompt supportive review, not an automatic adverse decision.

## 10. Conclusion
The project demonstrates the end-to-end AI/ML workflow from dataset preprocessing and model training to evaluation and Streamlit deployment. The expanded interface adds student record management, batch prediction, CSV export and PDF reporting.

## 11. Future Scope
- Use a documented real-world dataset with appropriate permissions.
- Apply cross-validation and hyperparameter tuning.
- Add authentication, encryption, audit logging and access controls before handling real student records.
- Assess model fairness and data representativeness.
- Consult educators and students before using predictions to guide interventions.

## 12. References
1. scikit-learn: https://scikit-learn.org/
2. Streamlit: https://docs.streamlit.io/
3. pandas: https://pandas.pydata.org/docs/
4. ReportLab: https://docs.reportlab.com/
