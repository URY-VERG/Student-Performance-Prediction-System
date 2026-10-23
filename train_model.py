from pathlib import Path
import json, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "student_performance.csv"
FEATURES = ["study_hours_per_day","attendance_percent","previous_score","assignment_average","sleep_hours","class_participation"]
TARGET = "final_score"

def main():
    df = pd.read_csv(DATA)
    X_train, X_test, y_train, y_test = train_test_split(df[FEATURES], df[TARGET], test_size=.2, random_state=42)
    models = {
        "Linear Regression": Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",LinearRegression())]),
        "Decision Tree": Pipeline([("imputer",SimpleImputer(strategy="median")),("model",DecisionTreeRegressor(max_depth=6,random_state=42))]),
        "Random Forest": Pipeline([("imputer",SimpleImputer(strategy="median")),("model",RandomForestRegressor(n_estimators=200,max_depth=10,random_state=42))])
    }
    scores={}
    best=None; best_r2=float("-inf")
    for name, model in models.items():
        model.fit(X_train,y_train)
        pred=model.predict(X_test)
        rmse=mean_squared_error(y_test,pred)**0.5
        scores[name]={"MAE":round(float(mean_absolute_error(y_test,pred)),3),"RMSE":round(float(rmse),3),"R2":round(float(r2_score(y_test,pred)),3)}
        if scores[name]["R2"]>best_r2: best,best_r2=name,scores[name]["R2"]
    joblib.dump({"models":models,"selected_model":best,"features":FEATURES},ROOT/"student_models.joblib")
    (ROOT/"model_metrics.json").write_text(json.dumps({"models":scores,"selected_model":best},indent=2),encoding="utf-8")
    print("Training complete. Selected model:",best)
    print(json.dumps(scores,indent=2))
    print("Saved student_models.joblib and model_metrics.json")
if __name__=="__main__": main()
study_hours_per_day
attendance_percent
previous_score
assignment_average
sleep_hours
class_participation
final_score