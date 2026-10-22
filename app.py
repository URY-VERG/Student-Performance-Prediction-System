from pathlib import Path
import io, json, sqlite3
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT=Path(__file__).resolve().parent
DATA_PATH=ROOT/"student_performance.csv"
MODEL_PATH=ROOT/"student_models.joblib"
METRICS_PATH=ROOT/"model_metrics.json"
DB_PATH=ROOT/"students.db"
FEATURES=["study_hours_per_day","attendance_percent","previous_score","assignment_average","sleep_hours","class_participation"]
TARGET="final_score"

st.set_page_config(page_title="Student Performance AI",page_icon="🎓",layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.6rem;max-width:1400px}
[data-testid="stMetric"]{background:rgba(100,130,180,.10);padding:14px;border-radius:12px;border:1px solid rgba(100,130,180,.2)}
.hero{padding:20px 24px;border-radius:16px;background:linear-gradient(120deg,#173b68,#356fa5);color:white;margin-bottom:18px}
.hero h1{color:white;margin:0}.hero p{margin:6px 0 0;color:#e6f0ff}
</style>
""",unsafe_allow_html=True)

def db_conn():
    conn=sqlite3.connect(DB_PATH,check_same_thread=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS students(
        id INTEGER PRIMARY KEY AUTOINCREMENT, student_name TEXT NOT NULL,
        student_id TEXT, department TEXT, semester TEXT,
        study_hours_per_day REAL, attendance_percent REAL, previous_score REAL,
        assignment_average REAL, sleep_hours REAL, class_participation INTEGER,
        predicted_score REAL, category TEXT, created_at TEXT)""")
    conn.commit()
    return conn

def fit_models():
    df=pd.read_csv(DATA_PATH)
    X_train,X_test,y_train,y_test=train_test_split(df[FEATURES],df[TARGET],test_size=.2,random_state=42)
    models={
      "Linear Regression":Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",LinearRegression())]),
      "Decision Tree":Pipeline([("imputer",SimpleImputer(strategy="median")),("model",DecisionTreeRegressor(max_depth=6,random_state=42))]),
      "Random Forest":Pipeline([("imputer",SimpleImputer(strategy="median")),("model",RandomForestRegressor(n_estimators=200,max_depth=10,random_state=42))])
    }
    metrics={}; best=None; best_r2=-float("inf")
    for name,model in models.items():
        model.fit(X_train,y_train); pred=model.predict(X_test)
        metrics[name]={"MAE":round(float(mean_absolute_error(y_test,pred)),3),
                       "RMSE":round(float(mean_squared_error(y_test,pred)**.5),3),
                       "R2":round(float(r2_score(y_test,pred)),3)}
        if metrics[name]["R2"]>best_r2: best,best_r2=name,metrics[name]["R2"]
    joblib.dump({"models":models,"selected_model":best,"features":FEATURES},MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({"models":metrics,"selected_model":best},indent=2),encoding="utf-8")
    return {"models":models,"selected_model":best,"features":FEATURES},{"models":metrics,"selected_model":best}

@st.cache_resource
def load_assets():
    if MODEL_PATH.exists() and METRICS_PATH.exists():
        return joblib.load(MODEL_PATH),json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return fit_models()

bundle,metrics=load_assets()
data=pd.read_csv(DATA_PATH)
conn=db_conn()
st.markdown('<div class="hero"><h1>🎓 Student Performance Prediction</h1><p>AI/ML-powered academic insights • Student records • Batch predictions • Reports</p></div>',unsafe_allow_html=True)
st.warning("Demonstration notice: included dataset is synthetic sample data. Predictions are estimates and should not be used as the sole basis for decisions about students.")

with st.sidebar:
    st.title("Navigation")
    page=st.radio("Open section",["Dashboard","Predict Student","Student Records","Batch CSV Prediction","Model Lab","About & Methodology"])
    st.divider()
    st.caption("AI/ML Mini Project • CO1–CO5")

def category_for(score):
    if score>=75: return "High"
    if score>=50: return "Moderate"
    return "Support recommended"

def advice_for(score,attendance,study):
    notes=[]
    if attendance<75: notes.append("Discuss attendance barriers and make a realistic attendance plan.")
    if study<2: notes.append("Try short, regular study sessions and weekly revision.")
    if score<50: notes.append("Offer supportive tutoring, practice questions, and teacher feedback.")
    elif score<75: notes.append("Use a weekly revision plan and practice weak topics.")
    else: notes.append("Maintain consistent habits and challenge learning with extension tasks.")
    return " ".join(notes)

def predict_row(row,model_name=None):
    name=model_name or bundle["selected_model"]
    vals=pd.DataFrame([{k:row[k] for k in FEATURES}],columns=FEATURES)
    score=float(bundle["models"][name].predict(vals)[0])
    return float(np.clip(score,0,100))

def pdf_report(name,student_id,department,semester,inputs,score,model):
    buf=io.BytesIO(); c=canvas.Canvas(buf,pagesize=A4); w,h=A4
    c.setTitle("Student Performance Prediction Report")
    c.setFont("Helvetica-Bold",18); c.drawString(48,h-55,"Student Performance Prediction Report")
    c.setFont("Helvetica",10); c.drawString(48,h-76,"AI/ML Mini Project • Generated "+datetime.now().strftime("%d-%m-%Y %H:%M"))
    c.line(48,h-88,w-48,h-88)
    y=h-120
    c.setFont("Helvetica-Bold",12); c.drawString(48,y,"Student Details"); y-=22
    c.setFont("Helvetica",11)
    for label,val in [("Name",name),("Student ID",student_id or "Not provided"),("Department",department or "Not provided"),("Semester",semester or "Not provided")]:
        c.drawString(55,y,f"{label}: {val}"); y-=18
    y-=8; c.setFont("Helvetica-Bold",12); c.drawString(48,y,"Input Features"); y-=22
    c.setFont("Helvetica",10)
    labels={"study_hours_per_day":"Study hours/day","attendance_percent":"Attendance (%)","previous_score":"Previous score","assignment_average":"Assignment average (%)","sleep_hours":"Sleep hours/night","class_participation":"Class participation (1-5)"}
    for k,v in inputs.items():
        c.drawString(55,y,f"{labels.get(k,k)}: {v}"); y-=16
    y-=10; c.setFont("Helvetica-Bold",14); c.drawString(48,y,f"Estimated final score: {score:.1f} / 100"); y-=22
    c.setFont("Helvetica",11); c.drawString(48,y,f"Performance band: {category_for(score)}"); y-=18
    c.drawString(48,y,f"Model: {model}"); y-=30
    c.setFont("Helvetica-Bold",11); c.drawString(48,y,"Suggested next step"); y-=18
    c.setFont("Helvetica",10)
    text=advice_for(score,inputs["attendance_percent"],inputs["study_hours_per_day"])
    for i in range(0,len(text),95): c.drawString(55,y,text[i:i+95]); y-=14
    y-=20; c.setFont("Helvetica-Oblique",9)
    c.drawString(48,y,"Important: this is an educational estimate based on a synthetic demonstration dataset.")
    c.drawString(48,y-14,"It is not a formal assessment or a substitute for teacher judgement.")
    c.save(); buf.seek(0); return buf.getvalue()

def input_form(prefix="p"):
    col1,col2,col3=st.columns(3)
    with col1:
        name=st.text_input("Student name",key=prefix+"name",value="")
        sid=st.text_input("Student ID (optional)",key=prefix+"id",value="")
        department=st.text_input("Department",key=prefix+"dept",value="IT")
    with col2:
        semester=st.selectbox("Semester",["1","2","3","4","5","6","7","8"],index=5,key=prefix+"sem")
        study=st.slider("Study hours/day",0.0,12.0,4.0,0.5,key=prefix+"study")
        attendance=st.slider("Attendance (%)",0.0,100.0,80.0,1.0,key=prefix+"att")
    with col3:
        previous=st.slider("Previous score",0.0,100.0,65.0,1.0,key=prefix+"prev")
        assignment=st.slider("Assignment average (%)",0.0,100.0,75.0,1.0,key=prefix+"assign")
        sleep=st.slider("Sleep hours/night",0.0,12.0,7.0,0.5,key=prefix+"sleep")
    participation=st.slider("Class participation (1–5)",1,5,3,key=prefix+"part")
    vals={"study_hours_per_day":study,"attendance_percent":attendance,"previous_score":previous,"assignment_average":assignment,"sleep_hours":sleep,"class_participation":participation}
    return name,sid,department,semester,vals

if page=="Dashboard":
    records=pd.read_sql_query("SELECT * FROM students",conn)
    total=len(records)
    avg=float(records["predicted_score"].mean()) if total else float(data[TARGET].mean())
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Student records",f"{total}")
    c2.metric("Sample dataset rows",f"{len(data):,}")
    c3.metric("Average predicted score" if total else "Sample average score",f"{avg:.1f}/100")
    c4.metric("Selected ML model",bundle["selected_model"])
    st.subheader("Academic Dataset Overview")
    g1,g2=st.columns(2)
    with g1:
        fig,ax=plt.subplots(); ax.hist(data[TARGET],bins=20); ax.set_xlabel("Final score"); ax.set_ylabel("Records"); st.pyplot(fig); plt.close(fig)
    with g2:
        fig,ax=plt.subplots(); ax.scatter(data["study_hours_per_day"],data[TARGET],alpha=.55); ax.set_xlabel("Study hours/day"); ax.set_ylabel("Final score"); st.pyplot(fig); plt.close(fig)
    if total:
        st.subheader("Recent saved predictions")
        st.dataframe(records.tail(10).iloc[::-1],use_container_width=True,hide_index=True)
    st.caption("The sample average is computed from synthetic data. Saved records are stored locally in students.db.")

elif page=="Predict Student":
    st.subheader("Individual Student Prediction")
    name,sid,dept,sem,vals=input_form("single_")
    model_name=st.selectbox("Prediction model",list(bundle["models"].keys()),index=list(bundle["models"].keys()).index(bundle["selected_model"]))
    if st.button("Predict and Save Student",type="primary"):
        if not name.strip():
            st.error("Please enter a student name before saving.")
        else:
            score=predict_row(vals,model_name); cat=category_for(score)
            conn.execute("""INSERT INTO students(student_name,student_id,department,semester,study_hours_per_day,attendance_percent,previous_score,assignment_average,sleep_hours,class_participation,predicted_score,category,created_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",(name.strip(),sid.strip(),dept,sem,*[vals[k] for k in FEATURES],score,cat,datetime.now().isoformat(timespec="seconds")))
            conn.commit()
            st.success("Prediction saved to the local student database.")
            a,b=st.columns(2); a.metric("Estimated final score",f"{score:.1f}/100"); b.metric("Performance band",cat)
            st.info(advice_for(score,vals["attendance_percent"],vals["study_hours_per_day"]))
            st.download_button("Download Student PDF Report",pdf_report(name,sid,dept,sem,vals,score,model_name),file_name=f"student_report_{datetime.now():%Y%m%d_%H%M%S}.pdf",mime="application/pdf")
            st.cache_data.clear()

elif page=="Student Records":
    st.subheader("Student Records")
    records=pd.read_sql_query("SELECT * FROM students ORDER BY id DESC",conn)
    if records.empty:
        st.info("No student records saved yet. Use Predict Student to add one.")
    else:
        search=st.text_input("Search by name, ID or department")
        filtered=records
        if search:
            mask=records.astype(str).apply(lambda col:col.str.contains(search,case=False,na=False)).any(axis=1)
            filtered=records[mask]
        st.dataframe(filtered,use_container_width=True,hide_index=True)
        st.download_button("Export records as CSV",filtered.to_csv(index=False).encode("utf-8"),"student_records.csv","text/csv")
        ids=filtered["id"].tolist()
        if ids:
            delete_id=st.selectbox("Select record ID to delete",ids)
            if st.button("Delete selected record"):
                conn.execute("DELETE FROM students WHERE id=?",(int(delete_id),)); conn.commit(); st.success("Record deleted. Refresh the page to update the table."); st.rerun()
        st.caption("Records are stored locally in students.db on this computer.")

elif page=="Batch CSV Prediction":
    st.subheader("Batch Prediction from CSV")
    st.write("Upload a CSV with these feature columns:")
    st.code(", ".join(FEATURES))
    uploaded=st.file_uploader("Upload student feature CSV",type=["csv"])
    model_name=st.selectbox("Model for batch prediction",list(bundle["models"].keys()),key="batchmodel")
    if uploaded:
        try:
            batch=pd.read_csv(uploaded)
            missing=[c for c in FEATURES if c not in batch.columns]
            if missing:
                st.error("Missing required columns: "+", ".join(missing))
            else:
                for c in FEATURES: batch[c]=pd.to_numeric(batch[c],errors="coerce")
                if batch[FEATURES].isna().all(axis=None):
                    st.error("No usable numeric feature values were found.")
                else:
                    batch["predicted_score"]=np.clip(bundle["models"][model_name].predict(batch[FEATURES]),0,100).round(1)
                    batch["performance_band"]=batch["predicted_score"].apply(category_for)
                    batch["support_flag"]=np.where((batch["predicted_score"]<50)|(batch["attendance_percent"]<75),"Review support needs","No automatic flag")
                    st.dataframe(batch,use_container_width=True)
                    st.download_button("Download predictions CSV",batch.to_csv(index=False).encode("utf-8"),"batch_predictions.csv","text/csv")
        except Exception as e: st.error(f"Could not process this CSV: {e}")
    st.download_button("Download blank CSV template",pd.DataFrame(columns=FEATURES).to_csv(index=False).encode("utf-8"),"student_input_template.csv","text/csv")

elif page=="Model Lab":
    st.subheader("Model Performance Comparison")
    comparison=pd.DataFrame(metrics["models"]).T
    st.dataframe(comparison,use_container_width=True)
    st.success(f"Selected model: {metrics['selected_model']}")
    st.write("MAE and RMSE: lower is better. R²: closer to 1 is generally better.")
    fig,ax=plt.subplots(); comparison[["MAE","RMSE"]].plot(kind="bar",ax=ax); ax.set_ylabel("Error"); ax.set_xlabel("Model"); st.pyplot(fig); plt.close(fig)
    fig,ax=plt.subplots(); comparison["R2"].plot(kind="bar",ax=ax); ax.set_ylabel("R²"); ax.set_xlabel("Model"); st.pyplot(fig); plt.close(fig)
    st.subheader("Model optimization")
    st.write("This baseline compares three models with fixed settings. For formal evaluation, use cross-validation and tune hyperparameters on training data, then evaluate once on a held-out test set.")
    st.download_button("Download metrics JSON",json.dumps(metrics,indent=2),"model_metrics.json","application/json")

else:
    st.subheader("About the Project")
    st.markdown("""
    **Problem definition:** estimate a final score from selected academic and study indicators.

    **Dataset collection:** the included `student_performance.csv` is synthetic demonstration data (500 records), not real student data.

    **Preprocessing:** feature/target separation, 80/20 train-test split, median imputation; feature scaling for Linear Regression.

    **Model development:** Linear Regression, Decision Tree Regressor and Random Forest Regressor.

    **Evaluation:** MAE, RMSE and R² on a held-out test set.

    **Optimization:** compare models and select the model with the highest test-set R² for the baseline. Further cross-validation and hyperparameter search are future improvements.

    **Result analysis:** dashboard plots, model metrics, predictions, CSV batch output and PDF report.

    **Ethics and limits:** a prediction is only an estimate. Use permissioned, representative data, protect student privacy, evaluate fairness, and involve educators before real-world use.
    """)
    st.subheader("Dataset preview")
    st.dataframe(data.head(15),use_container_width=True,hide_index=True)
    st.download_button("Download source dataset CSV",data.to_csv(index=False).encode("utf-8"),"student_performance.csv","text/csv")
