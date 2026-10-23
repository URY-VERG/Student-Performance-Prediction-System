from pathlib import Path
import io, json, sqlite3
import re
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

st.set_page_config(page_title="Student Performance Intelligence",page_icon="🎓",layout="wide")
st.markdown("""
<style>
.stApp{background:radial-gradient(ellipse at 8% 8%,rgba(99,102,241,.10),transparent 28%),radial-gradient(ellipse at 95% 18%,rgba(34,211,238,.10),transparent 30%),linear-gradient(145deg,#f4f6fc 0%,#f8f7ff 48%,#f1fbfa 100%);color:#192944}
.block-container{padding-top:1.5rem;padding-bottom:2.5rem;max-width:1480px}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1528 0%,#172554 52%,#123c52 100%)}
[data-testid="stSidebar"] *{color:#f8fbff}
[data-testid="stSidebar"] [data-testid="stRadio"] label{border-radius:10px;padding:5px 8px;transition:background .2s ease}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover{background:rgba(255,255,255,.13)}
.hero{position:relative;overflow:hidden;padding:30px 34px;border:1px solid rgba(255,255,255,.3);border-radius:24px;background:linear-gradient(115deg,#101b34 0%,#293b83 48%,#087e8b 100%);color:white;margin-bottom:20px;box-shadow:0 16px 36px rgba(21,34,76,.22)}
.hero:after{content:"AI  /  EDU";position:absolute;right:32px;bottom:10px;font-size:25px;font-weight:800;opacity:.20;letter-spacing:5px}
.hero h1{color:white!important;margin:0;font-size:clamp(1.8rem,3vw,2.5rem);font-weight:850;letter-spacing:-.6px}
.hero p{margin:8px 0 0;color:#e8f5ff;font-size:1rem}
h1,h2,h3{color:#1e2e58}
h2{padding-bottom:7px;border-bottom:2px solid transparent;border-image:linear-gradient(90deg,#6366f1,#14b8a6,#f59e0b) 1}
h3{color:#334780}
[data-testid="stMetric"]{background:linear-gradient(145deg,#fff 0%,#f7f9ff 100%);padding:18px 20px;border-radius:18px;border:1px solid #e0e7ff;box-shadow:0 7px 20px rgba(41,57,112,.08);transition:transform .2s ease,box-shadow .2s ease}
[data-testid="stMetric"]:hover{transform:translateY(-3px);box-shadow:0 12px 25px rgba(41,57,112,.14)}
[data-testid="stMetricLabel"]{color:#52648a}
.section-kicker{color:#637393;font-size:.78rem;font-weight:750;letter-spacing:.12em;text-transform:uppercase;margin:0 0 4px}
.empty-state{padding:24px;border:1px dashed #aab8d3;border-radius:16px;background:rgba(255,255,255,.65);color:#52648a;text-align:center}
.app-footer{margin-top:42px;padding:18px 4px;border-top:1px solid #dce5f5;color:#65738d;font-size:.82rem}
.stButton>button,[data-testid="stDownloadButton"]>button{background:linear-gradient(100deg,#4f46e5,#2563eb 55%,#0d9488);color:white;border:0;border-radius:12px;font-weight:700;padding:.62rem 1.1rem;box-shadow:0 5px 13px rgba(59,78,190,.18);transition:transform .2s ease,box-shadow .2s ease}
.stButton>button:hover,[data-testid="stDownloadButton"]>button:hover{color:white;border:0;transform:translateY(-2px);box-shadow:0 8px 18px rgba(59,78,190,.28)}
[data-testid="stDataFrame"],[data-testid="stTable"]{background:white;border:1px solid #e3e9f5;border-radius:14px;overflow:hidden}
[data-testid="stFileUploader"]{background:rgba(255,255,255,.78);border:1px dashed #8b9be9;border-radius:16px;padding:12px}
[data-testid="stAlert"]{border-radius:14px}
[data-baseweb="input"]>div,[data-baseweb="select"]>div{border-radius:10px}
div[data-testid="stSlider"] [data-baseweb="slider"]{padding-top:5px}
hr{border-color:#dce5f5}
@media(max-width:700px){.hero{padding:23px 20px}.hero:after{font-size:22px;right:15px}.block-container{padding-left:1rem;padding-right:1rem}}
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
    metrics_data={"models":{},"selected_model":None}
    if METRICS_PATH.exists():
        try:
            metrics_data=json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError):
            metrics_data={"models":{},"selected_model":None}
    if not MODEL_PATH.exists():
        return {"models":{},"selected_model":None,"features":FEATURES},metrics_data
    try:
        saved_bundle=joblib.load(MODEL_PATH)
        if not isinstance(saved_bundle,dict) or not isinstance(saved_bundle.get("models"),dict):
            raise ValueError("Saved model artifact has an unsupported structure.")
        if saved_bundle.get("features",FEATURES)!=FEATURES:
            raise ValueError("Saved model feature order does not match the current prediction form.")
        required_models={"Linear Regression","Decision Tree","Random Forest"}
        missing_models=required_models-set(saved_bundle["models"])
        if missing_models:
            raise ValueError("Saved model artifact is missing: "+", ".join(sorted(missing_models)))
        return saved_bundle,metrics_data
    except Exception as exc:
        return {"models":{},"selected_model":None,"features":FEATURES},{**metrics_data,"load_error":str(exc)}

if not DATA_PATH.exists():
    st.error(f"Training dataset not found: {DATA_PATH.name}. Restore the project CSV to continue.")
    st.stop()
bundle,metrics=load_assets()
data=pd.read_csv(DATA_PATH)
conn=db_conn()
st.markdown('<div class="hero"><h1>◈ Student Performance Intelligence</h1><p>AI-Powered Academic Analytics &amp; Prediction &nbsp; · &nbsp; Model insights, student support, and college result analysis</p></div>',unsafe_allow_html=True)
st.warning("Demonstration data notice: the included dataset is synthetic. Scores are model estimates, not actual student results, and must not be used as the sole basis for student decisions.")
if not bundle.get("models"):
    st.error("A compatible saved model could not be loaded. The app will not retrain automatically. Restore student_models.joblib or run train_model.py intentionally to create model artifacts.")

with st.sidebar:
    st.title("Student Intelligence")
    page = st.radio(
        "✨ Choose a section",
        [
            "Dashboard",
            "College Result Analysis",
            "Predict Student",
            "Student Records",
            "Batch CSV Prediction",
            "At-Risk Alerts",
            "Model Lab",
            "Reports",
            "Settings & Methodology"
        ]
    )
    st.divider()
    st.caption("College Edition · AI/ML academic analytics")

st.markdown(f'<div class="section-kicker">CURRENT WORKSPACE &nbsp; / &nbsp; {page}</div>',unsafe_allow_html=True)

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

def normalize_column(column):
    text = str(column).replace("\ufeff", "").replace("\n", " ").strip().lower()
    return re.sub(r"\s+", " ", text)

def read_generic_college_file(uploaded_file):
    """Read a generic uploaded result into a dataframe for preview or column mapping."""
    content = uploaded_file.getvalue()
    if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(content))
    return pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")

def parse_college_result(uploaded_file, column_mapping=None):
    """Read the supported college workbook layout or a generic Excel/CSV file."""
    filename = uploaded_file.name.lower()
    if filename.endswith((".xlsx", ".xls")):
        raw = pd.read_excel(uploaded_file, header=None)
        if raw.shape[1] >= 25 and str(raw.iloc[0, 2]).strip().lower().startswith("full name"):
            records = []
            subjects = ["ISR", "SPM", "DL", "MT", "WC"]
            for _, row in raw.iloc[2:].iterrows():
                name = row.iloc[2]
                if pd.isna(name) or str(name).strip().lower() in {"", "nan"}:
                    continue
                record = {
                    "Roll No": row.iloc[0],
                    "Student ID": row.iloc[1],
                    "Student Name": str(name).strip(),
                    "SGPA": pd.to_numeric(row.iloc[23], errors="coerce"),
                    "Class": str(row.iloc[24]).strip() if pd.notna(row.iloc[24]) else "Not recorded",
                }
                for index, subject in enumerate(subjects):
                    record[subject] = pd.to_numeric(row.iloc[13 + index], errors="coerce")
                records.append(record)
            return pd.DataFrame(records), "College Excel result sheet"
        df = read_generic_college_file(uploaded_file)
    else:
        df = read_generic_college_file(uploaded_file)

    original_columns = list(df.columns)
    aliases = {normalize_column(column): column for column in original_columns}
    name_aliases = ["student name", "name", "full name", "full name of the candidate", "candidate name", "student"]
    mapped_name = (column_mapping or {}).get("Student Name")
    name_column = mapped_name if mapped_name in original_columns else next((aliases[key] for key in name_aliases if key in aliases), None)
    if name_column is None:
        raise ValueError("Student Name column not found. Include a Student Name, Name, or Full Name column.")

    result = pd.DataFrame({"Student Name": df[name_column].astype(str).str.strip()})
    id_aliases = ["student id", "seat number", "exam seat number", "roll no", "roll number"]
    mapped_id = (column_mapping or {}).get("Student ID")
    id_column = mapped_id if mapped_id in original_columns else next((aliases[key] for key in id_aliases if key in aliases), None)
    if id_column is not None:
        result["Student ID"] = df[id_column]

    subject_names = {"ISR", "SPM", "DL", "MT", "WC"}
    mapped_columns = {source: target for target, source in (column_mapping or {}).items() if source in original_columns and target != "Student Name" and target != "Student ID"}
    for column in original_columns:
        normalized = normalize_column(column)
        upper = str(column).strip().upper()
        if column in mapped_columns or normalized in {"sgpa", "cgpa", "class", "result", "grade"} or upper in subject_names:
            output_column = mapped_columns.get(column, upper if upper in subject_names else normalized.title())
            result[output_column] = df[column]

    for column in result.columns:
        if column not in {"Student Name", "Student ID", "Class", "Result", "Grade"}:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result, "Uploaded result file"

def college_suggestion(row, subject_columns):
    weak = [column for column in subject_columns if pd.notna(row.get(column)) and float(row[column]) < 50]
    if weak:
        return "Focus on " + ", ".join(weak) + ": revise fundamentals, practice regularly, and ask the subject teacher for feedback."
    known = [column for column in subject_columns if pd.notna(row.get(column))]
    if known and all(float(row[column]) >= 75 for column in known):
        return "Strong subject performance. Maintain revision and attempt advanced problems or peer mentoring."
    return "Create a weekly revision timetable, practice previous papers, and review mistakes after each test."

def risk_level_for(score, attendance, high_threshold, moderate_threshold, attendance_threshold):
    if score < high_threshold:
        return "High"
    if score < moderate_threshold or attendance < attendance_threshold:
        return "Moderate"
    return "Low"

def dataframe_pdf_report(title, frame, methodology_note):
    """Create a compact, paginated PDF table from the selected real data."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    page_width, page_height = A4
    margin = 42
    y = page_height - 48
    pdf.setTitle(title)
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(margin, y, title[:75])
    y -= 18
    pdf.setFont("Helvetica", 9)
    pdf.drawString(margin, y, f"Generated {datetime.now():%Y-%m-%d %H:%M} | {methodology_note[:100]}")
    y -= 22
    columns = [str(column) for column in frame.columns[:6]]
    rows = frame[columns].head(100).astype(str).values.tolist() if columns else []
    col_width = (page_width - 2 * margin) / max(len(columns), 1)

    def printable(value):
        return str(value).encode("latin-1", errors="replace").decode("latin-1")[:22]

    pdf.setFont("Helvetica-Bold", 8)
    for index, column in enumerate(columns):
        pdf.drawString(margin + index * col_width, y, printable(column))
    y -= 13
    pdf.setFont("Helvetica", 8)
    for row in rows:
        if y < 48:
            pdf.showPage()
            y = page_height - 48
            pdf.setFont("Helvetica", 8)
        for index, value in enumerate(row):
            pdf.drawString(margin + index * col_width, y, printable(value))
        y -= 12
    if len(frame) > 100:
        pdf.drawString(margin, max(y - 8, 35), "PDF preview limited to first 100 rows. Download CSV for the complete dataset.")
    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()

if page=="Dashboard":
    records=pd.read_sql_query("SELECT * FROM students ORDER BY id DESC",conn)
    total=len(records)
    risk_threshold=st.session_state.get("risk_high_threshold",50)
    moderate_threshold=st.session_state.get("risk_moderate_threshold",65)
    attendance_threshold=st.session_state.get("risk_attendance_threshold",75)
    if total:
        records["risk_level"]=records.apply(lambda row:risk_level_for(row["predicted_score"],row["attendance_percent"],risk_threshold,moderate_threshold,attendance_threshold),axis=1)
    high_count=int((records["predicted_score"]>=75).sum()) if total else 0
    at_risk_count=int((records["risk_level"]!="Low").sum()) if total else 0
    avg_prediction=float(records["predicted_score"].mean()) if total else None
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("Saved predictions",f"{total:,}")
    c2.metric("Avg. predicted score",f"{avg_prediction:.1f}/100" if avg_prediction is not None else "No saved predictions")
    c3.metric("At-risk review",f"{at_risk_count:,}")
    c4.metric("High predicted band",f"{high_count:,}")
    c5.metric("Selected model",bundle.get("selected_model") or "Unavailable")
    st.caption(f"Saved predictions: estimates from the selected model. Sample training target average: {data[TARGET].mean():.1f}/100 across {len(data):,} synthetic rows; this is separate from saved predictions.")
    if total:
        departments=sorted(records["department"].dropna().astype(str).unique())
        band_options=["All bands","High","Moderate","Support recommended"]
        f1,f2,f3=st.columns(3)
        chosen_departments=f1.multiselect("Filter department",departments,placeholder="All departments")
        chosen_bands=f2.multiselect("Filter predicted band",band_options[1:],placeholder="All bands")
        activity_window=f3.selectbox("Prediction activity",["All dates","Last 30 days","Last 7 days"],key="dashboard_activity_window")
        filtered_records=records.copy()
        if chosen_departments:
            filtered_records=filtered_records[filtered_records["department"].astype(str).isin(chosen_departments)]
        if chosen_bands:
            filtered_records=filtered_records[filtered_records["category"].isin(chosen_bands)]
        if activity_window!="All dates":
            days=30 if activity_window=="Last 30 days" else 7
            cutoff=pd.Timestamp.now()-pd.Timedelta(days=days)
            created=pd.to_datetime(filtered_records["created_at"],errors="coerce")
            filtered_records=filtered_records[created>=cutoff]
    else:
        filtered_records=records
    st.subheader("📊 Academic Dataset Overview")
    g1,g2=st.columns(2)
    with g1:
        fig,ax=plt.subplots(); ax.hist(data[TARGET].dropna(),bins=20,color="#6366f1",edgecolor="white"); ax.set_xlabel("Synthetic training target score"); ax.set_ylabel("Dataset rows"); ax.set_title("Training dataset target distribution (not saved predictions)"); st.pyplot(fig); plt.close(fig)
    with g2:
        fig,ax=plt.subplots(); ax.scatter(data["study_hours_per_day"],data[TARGET],alpha=.6,color="#0d9488",edgecolors="white"); ax.set_xlabel("Study hours per day"); ax.set_ylabel("Synthetic final score"); ax.set_title("Study indicator vs synthetic target"); st.pyplot(fig); plt.close(fig)
    left,right=st.columns(2)
    with left:
        st.markdown("#### Model evaluation snapshot")
        selected_metrics=metrics.get("models",{}).get(metrics.get("selected_model",bundle["selected_model"]),{})
        if selected_metrics:
            a,b,c=st.columns(3)
            a.metric("MAE",selected_metrics.get("MAE","N/A"))
            b.metric("RMSE",selected_metrics.get("RMSE","N/A"))
            c.metric("R²",selected_metrics.get("R2","N/A"))
            st.caption("Held-out synthetic test split; metrics are not real-world accuracy guarantees.")
        else:
            st.info("Model metrics are unavailable in the saved metrics file.")
    with right:
        st.markdown("#### Saved prediction bands")
        if total:
            band_counts=records["category"].value_counts()
            fig,ax=plt.subplots(); band_counts.plot(kind="bar",ax=ax,color=["#14b8a6","#6366f1","#f97316"]); ax.set_ylabel("Saved predictions"); ax.set_xlabel("Predicted band"); ax.tick_params(axis="x",rotation=15); st.pyplot(fig); plt.close(fig)
        else:
            st.markdown('<div class="empty-state">No saved predictions yet. Add one in Predict Student to begin activity tracking.</div>',unsafe_allow_html=True)
    st.subheader("🕘 Recent saved prediction activity")
    if total:
        st.dataframe(filtered_records.head(10),use_container_width=True,hide_index=True)
    else:
        st.caption("Student prediction records are stored locally in students.db. No activity is recorded yet.")
    if total:
        st.info(f"Current saved records: {at_risk_count:,} meet at least one configured review threshold; {high_count:,} have a High predicted band. These are estimates based only on local saved prediction data.")

elif page=="College Result Analysis":
    st.subheader("🏫 College Result Intelligence")
    st.caption("Upload semester results to review class distribution, subject marks, and students who may benefit from support.")
    result_file=st.file_uploader("📂 Upload college result (Excel or CSV)",type=["xlsx","xls","csv"],key="college_result_upload")
    if result_file:
        try:
            saved_mapping=st.session_state.get("college_result_column_mapping",{})
            if saved_mapping.get("_filename")!=result_file.name:
                saved_mapping={}
            result_df,source_label=parse_college_result(result_file,saved_mapping)
            if result_df.empty:
                st.warning("No student rows were found in this file.")
            else:
                st.success(f"Loaded {len(result_df)} student records • Source: {source_label}")
                st.session_state["latest_college_results"]=result_df.copy()
                with st.expander("👀 Preview uploaded result",expanded=False):
                    st.dataframe(result_df.head(10),use_container_width=True,hide_index=True)
                subject_cols=[column for column in ["ISR","SPM","DL","MT","WC"] if column in result_df.columns]
                if not subject_cols:
                    excluded={"Roll No","Student ID","Student Name","SGPA","Class","Result","Grade"}
                    subject_cols=[column for column in result_df.columns if column not in excluded and pd.api.types.is_numeric_dtype(result_df[column])]
                for column in subject_cols:
                    result_df[column]=pd.to_numeric(result_df[column],errors="coerce")
                if "SGPA" in result_df.columns:
                    result_df["SGPA"]=pd.to_numeric(result_df["SGPA"],errors="coerce")
                weak_threshold=st.slider("Weak-subject review threshold (configurable; not an official pass mark)",0.0,100.0,50.0,1.0,key="college_weak_threshold")
                invalid_mark_count=int(((result_df[subject_cols]<0)|(result_df[subject_cols]>100)).sum().sum()) if subject_cols else 0
                if invalid_mark_count:
                    st.warning(f"Found {invalid_mark_count} subject mark value(s) outside 0–100. Please verify the source file; these rows are still shown for review.")
                duplicate_subset=[column for column in ["Student ID","Student Name"] if column in result_df.columns]
                duplicate_count=int(result_df.duplicated(subset=duplicate_subset,keep=False).sum()) if duplicate_subset else 0
                if duplicate_count:
                    st.warning(f"{duplicate_count} potentially duplicate student rows detected. Review IDs/names before using this analysis.")
                result_df["Low Marks Subjects"]=result_df.apply(
                    lambda row:", ".join([column for column in subject_cols if pd.notna(row.get(column)) and float(row[column])<weak_threshold]) or "—",axis=1
                )
                result_df["AI Improvement Suggestions"]=result_df.apply(lambda row:college_suggestion(row,subject_cols),axis=1)
                if subject_cols:
                    result_df["Subject Marks Average"]=result_df[subject_cols].mean(axis=1).round(1)

                c1,c2,c3,c4=st.columns(4)
                c1.metric("👩‍🎓 Students",f"{len(result_df):,}")
                c2.metric("🆘 Need subject support",f"{int((result_df['Low Marks Subjects']!='—').sum()):,}")
                if "SGPA" in result_df.columns and result_df["SGPA"].notna().any():
                    c3.metric("📈 Average SGPA",f"{result_df['SGPA'].mean():.2f}")
                elif subject_cols:
                    c3.metric("📈 Average subject marks",f"{result_df[subject_cols].mean().mean():.1f}")
                else:
                    c3.metric("📈 Average","N/A")
                if "Class" in result_df.columns:
                    class_counts=result_df["Class"].fillna("Not recorded").astype(str).value_counts()
                    c4.metric("🏆 Top class group",str(class_counts.index[0])[:22] if len(class_counts) else "N/A")
                else:
                    c4.metric("📚 Subjects detected",str(len(subject_cols)))

                if "Result" in result_df.columns:
                    result_values=result_df["Result"].astype(str).str.strip().str.lower()
                    pass_count=int(result_values.isin(["pass","passed","p"]).sum())
                    fail_count=int(result_values.isin(["fail","failed","f"]).sum())
                    if pass_count+fail_count:
                        c1,c2,c3=st.columns(3)
                        c1.metric("✅ Recorded passes",f"{pass_count:,}")
                        c2.metric("❌ Recorded fails",f"{fail_count:,}")
                        c3.metric("📊 Pass percentage",f"{100*pass_count/(pass_count+fail_count):.1f}%")
                if "SGPA" in result_df.columns and result_df["SGPA"].notna().any():
                    st.markdown("#### 🥇 Top SGPA records")
                    st.dataframe(result_df.nlargest(5,"SGPA")[[column for column in ["Student Name","Student ID","SGPA","Class"] if column in result_df.columns]],use_container_width=True,hide_index=True)

                st.markdown("### 📊 Class overview")
                left,right=st.columns(2)
                with left:
                    if "Class" in result_df.columns:
                        counts=result_df["Class"].fillna("Not recorded").astype(str).value_counts()
                        fig,ax=plt.subplots(); counts.plot(kind="bar",ax=ax); ax.set_ylabel("Students"); ax.set_xlabel("Recorded class"); ax.tick_params(axis="x",rotation=30); st.pyplot(fig); plt.close(fig)
                    elif "Grade" in result_df.columns:
                        grades=result_df["Grade"].fillna("Not recorded").astype(str).value_counts()
                        fig,ax=plt.subplots(); grades.plot(kind="bar",ax=ax,color="#6366f1"); ax.set_ylabel("Students"); ax.set_xlabel("Recorded grade"); ax.tick_params(axis="x",rotation=25); st.pyplot(fig); plt.close(fig)
                    elif subject_cols:
                        means=result_df[subject_cols].mean().sort_values(ascending=False)
                        fig,ax=plt.subplots(); means.plot(kind="bar",ax=ax); ax.set_ylabel("Average marks"); ax.set_xlabel("Subject"); st.pyplot(fig); plt.close(fig)
                with right:
                    if subject_cols:
                        low_by_subject=(result_df[subject_cols]<weak_threshold).sum().sort_values(ascending=False)
                        fig,ax=plt.subplots(); low_by_subject.plot(kind="bar",ax=ax,color="#f97316"); ax.set_ylabel(f"Students below {weak_threshold:g} (review threshold)"); ax.set_xlabel("Subject"); st.pyplot(fig); plt.close(fig)
                    elif "SGPA" in result_df.columns:
                        fig,ax=plt.subplots(); result_df["SGPA"].dropna().plot(kind="hist",bins=10,ax=ax); ax.set_xlabel("SGPA"); st.pyplot(fig); plt.close(fig)
                if "SGPA" in result_df.columns and result_df["SGPA"].notna().any() and subject_cols:
                    with st.expander("📈 SGPA distribution"):
                        fig,ax=plt.subplots(); result_df["SGPA"].dropna().plot(kind="hist",bins=10,ax=ax,color="#8b5cf6",edgecolor="white"); ax.set_xlabel("Recorded SGPA"); ax.set_ylabel("Students"); st.pyplot(fig); plt.close(fig)

                st.markdown("### 🚨 Students who may need support")
                support=result_df[result_df["Low Marks Subjects"]!="—"]
                if support.empty:
                    st.success("No subject marks below 50 were detected with the current support threshold.")
                else:
                    st.dataframe(support,use_container_width=True,hide_index=True)
                if subject_cols:
                    st.markdown("### 🧠 Subject-wise improvement plan")
                    selected_subject=st.selectbox("Choose subject",subject_cols)
                    subject_mean=result_df[selected_subject].mean()
                    st.info(f"{selected_subject}: class average = {subject_mean:.1f}. Consider concept revision, guided practice, and a weekly mini-test for students below 50.")
                st.caption("Review thresholds are configurable starting points, not official pass/fail rules. Only explicit pass/fail values in the uploaded Result column are reported as recorded outcomes.")
                student_search=st.text_input("🔎 Search student results",key="college_student_search")
                visible_results=result_df
                if student_search:
                    visible_results=result_df[result_df.astype(str).apply(lambda column:column.str.contains(student_search,case=False,na=False,regex=False)).any(axis=1)]
                st.dataframe(visible_results,use_container_width=True,hide_index=True)
                st.download_button("⬇️ Download analyzed student report (CSV)",result_df.to_csv(index=False).encode("utf-8-sig"),file_name="college_student_analysis.csv",mime="text/csv")
                st.download_button("📄 Download college analysis PDF",dataframe_pdf_report("College Result Analysis",result_df,"Descriptive analysis of uploaded result data; check official college grading rules."),file_name="college_result_analysis.pdf",mime="application/pdf")
        except Exception as exc:
            st.error(f"Could not read this result file: {exc}")
            st.info("Use the mapping controls below when the student-name or result headers differ from the supported names.")
            try:
                raw_result=read_generic_college_file(result_file)
                source_columns=[str(column) for column in raw_result.columns]
                with st.form("college_column_mapping_form"):
                    st.markdown("##### Map uploaded columns")
                    name_index=st.selectbox("Student name column",source_columns,key="college_map_name")
                    id_options=["Not available",*source_columns]
                    id_choice=st.selectbox("Student ID column (optional)",id_options,key="college_map_id")
                    map_choices={"SGPA":"College result SGPA","Class":"College result class","Result":"Pass/fail result","Grade":"Grade"}
                    for subject in ["ISR","SPM","DL","MT","WC"]:
                        map_choices[subject]=f"Subject {subject} marks"
                    for standard_name,label in map_choices.items():
                        map_choices[standard_name]=st.selectbox(f"{label} (optional)",["Not available",*source_columns],key=f"college_map_{standard_name}")
                    apply_mapping=st.form_submit_button("Apply mapping and analyze")
                if apply_mapping:
                    new_mapping={"Student Name":name_index,"_filename":result_file.name}
                    if id_choice!="Not available":
                        new_mapping["Student ID"]=id_choice
                    for standard_name in map_choices:
                        selected_column=st.session_state.get(f"college_map_{standard_name}","Not available")
                        if selected_column!="Not available":
                            new_mapping[standard_name]=selected_column
                    st.session_state["college_result_column_mapping"]=new_mapping
                    st.rerun()
                st.dataframe(raw_result.head(8),use_container_width=True,hide_index=True)
            except Exception as mapping_error:
                st.error(f"Unable to preview this format: {mapping_error}")
    else:
        st.info("Upload a college result Excel/CSV to see student marks, summaries, weak subjects, and suggestions.")

elif page=="Predict Student":
    st.subheader("🔮 Individual Student Prediction")
    name,sid,dept,sem,vals=input_form("single_")
    available_models=list(bundle.get("models",{}))
    model_options=available_models or ["Model unavailable"]
    default_model=bundle.get("selected_model")
    model_index=available_models.index(default_model) if default_model in available_models else 0
    model_name=st.selectbox("Prediction model",model_options,index=model_index,disabled=not available_models)
    if st.button("✨ Predict and Save Student",type="primary",disabled=not available_models):
        if not name.strip():
            st.error("Please enter a student name before saving.")
        else:
            duplicate=conn.execute("SELECT id FROM students WHERE student_id=? AND TRIM(student_id)<>''",(sid.strip(),)).fetchone() if sid.strip() else None
            if duplicate:
                st.error(f"Student ID already exists on record #{duplicate[0]}. Use Student Records to review or edit that record.")
            else:
                score=predict_row(vals,model_name); cat=category_for(score)
                conn.execute("""INSERT INTO students(student_name,student_id,department,semester,study_hours_per_day,attendance_percent,previous_score,assignment_average,sleep_hours,class_participation,predicted_score,category,created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",(name.strip(),sid.strip(),dept,sem,*[vals[k] for k in FEATURES],score,cat,datetime.now().isoformat(timespec="seconds")))
                conn.commit()
                st.success("Prediction saved to the local student database.")
                a,b=st.columns(2); a.metric("Model-estimated score",f"{score:.1f}/100"); b.metric("Predicted band",cat)
                st.info(advice_for(score,vals["attendance_percent"],vals["study_hours_per_day"]))
                if model_name=="Linear Regression":
                    pipeline=bundle["models"][model_name]
                    scaler=pipeline.named_steps["scale"]
                    estimator=pipeline.named_steps["model"]
                    centered_values=np.array([vals[feature] for feature in FEATURES],dtype=float)
                    contributions=np.asarray(estimator.coef_)*(centered_values-scaler.mean_)/scaler.scale_
                    contribution_frame=pd.DataFrame({"Feature":[feature.replace("_"," ").title() for feature in FEATURES],"Contribution relative to training mean":contributions}).sort_values("Contribution relative to training mean")
                    with st.expander("How this linear estimate is composed"):
                        st.caption("Signed linear-model contributions relative to the training-data feature means. These are model explanations, not causal effects or guarantees.")
                        fig,ax=plt.subplots(); ax.barh(contribution_frame["Feature"],contribution_frame["Contribution relative to training mean"],color=["#f97316" if value<0 else "#0d9488" for value in contribution_frame["Contribution relative to training mean"]]); ax.axvline(0,color="#475569",linewidth=1); ax.set_xlabel("Score contribution (model units)"); st.pyplot(fig); plt.close(fig)
                st.download_button("📄 Download Student PDF Report",pdf_report(name,sid,dept,sem,vals,score,model_name),file_name=f"student_report_{datetime.now():%Y%m%d_%H%M%S}.pdf",mime="application/pdf")

elif page=="Student Records":
    st.subheader("🗂️ Student Records")
    records=pd.read_sql_query("SELECT * FROM students ORDER BY id DESC",conn)
    if records.empty:
        st.markdown('<div class="empty-state">No saved student predictions yet. Create a model estimate from the Predict Student section to begin a local activity history.</div>',unsafe_allow_html=True)
    else:
        search=st.text_input("🔎 Search by name, ID or department")
        departments=sorted(records["department"].dropna().astype(str).unique())
        bands=sorted(records["category"].dropna().astype(str).unique())
        filter1,filter2=st.columns(2)
        selected_departments=filter1.multiselect("Department",departments,placeholder="All departments",key="records_departments")
        selected_bands=filter2.multiselect("Predicted band",bands,placeholder="All bands",key="records_bands")
        filtered=records
        if search:
            mask=records.astype(str).apply(lambda col:col.str.contains(search,case=False,na=False,regex=False)).any(axis=1)
            filtered=records[mask]
        if selected_departments:
            filtered=filtered[filtered["department"].astype(str).isin(selected_departments)]
        if selected_bands:
            filtered=filtered[filtered["category"].isin(selected_bands)]
        page_size=st.selectbox("Rows per page",[10,25,50,100],index=1,key="records_page_size")
        page_count=max(1,(len(filtered)+page_size-1)//page_size)
        page_number=st.number_input("Page",min_value=1,max_value=page_count,value=1,step=1,key="records_page_number")
        first=(page_number-1)*page_size
        page_records=filtered.iloc[first:first+page_size]
        st.caption(f"Showing {len(page_records)} of {len(filtered):,} filtered records · {len(records):,} total")
        st.dataframe(page_records,use_container_width=True,hide_index=True)
        st.download_button("📥 Export filtered records as CSV",filtered.to_csv(index=False).encode("utf-8-sig"),"student_records.csv","text/csv")
        ids=filtered["id"].tolist()
        if ids:
            with st.expander("👤 View or edit a student profile"):
                record_id=st.selectbox("Choose a record",ids,format_func=lambda record_id:f"#{record_id} · {records.loc[records['id']==record_id,'student_name'].iloc[0]}",key="profile_record_id")
                record=records.loc[records["id"]==record_id].iloc[0]
                st.caption(f"Prediction recorded: {record['created_at']} · Existing model-estimated score: {record['predicted_score']:.1f}/100")
                with st.form(f"edit_student_{record_id}"):
                    p1,p2,p3=st.columns(3)
                    edit_name=p1.text_input("Student name",str(record["student_name"]),key=f"edit_name_{record_id}")
                    edit_id=p1.text_input("Student ID",str(record["student_id"] or ""),key=f"edit_id_{record_id}")
                    edit_department=p2.text_input("Department",str(record["department"] or ""),key=f"edit_department_{record_id}")
                    edit_semester=p2.text_input("Semester",str(record["semester"] or ""),key=f"edit_semester_{record_id}")
                    edit_models=list(bundle.get("models",{})) or ["Model unavailable"]
                    edit_selected=bundle.get("selected_model")
                    edit_index=edit_models.index(edit_selected) if edit_selected in edit_models else 0
                    edit_model=p3.selectbox("Recalculate estimate using",edit_models,index=edit_index,key=f"edit_model_{record_id}",disabled=not bundle.get("models"))
                    edited_values={}
                    feature_columns=st.columns(3)
                    for index,feature in enumerate(FEATURES):
                        label=feature.replace("_"," ").title()
                        bounds={"study_hours_per_day":(0.0,12.0),"attendance_percent":(0.0,100.0),"previous_score":(0.0,100.0),"assignment_average":(0.0,100.0),"sleep_hours":(0.0,12.0),"class_participation":(1.0,5.0)}
                        minimum,maximum=bounds[feature]
                        edited_values[feature]=feature_columns[index%3].number_input(label,min_value=minimum,max_value=maximum,value=min(max(float(record[feature] if pd.notna(record[feature]) else minimum),minimum),maximum),step=1.0 if feature not in {"study_hours_per_day","sleep_hours"} else .5,key=f"edit_{feature}_{record_id}")
                    save_edit=st.form_submit_button("Save profile and recalculate prediction",disabled=not bundle.get("models"))
                if save_edit:
                    duplicate=conn.execute("SELECT id FROM students WHERE student_id=? AND id<>? AND TRIM(student_id)<>''",(edit_id.strip(),int(record_id))).fetchone() if edit_id.strip() else None
                    if not edit_name.strip():
                        st.error("Student name is required.")
                    elif duplicate:
                        st.error(f"Student ID already exists on record #{duplicate[0]}.")
                    else:
                        edited_score=predict_row(edited_values,edit_model)
                        edited_category=category_for(edited_score)
                        conn.execute("""UPDATE students SET student_name=?,student_id=?,department=?,semester=?,study_hours_per_day=?,attendance_percent=?,previous_score=?,assignment_average=?,sleep_hours=?,class_participation=?,predicted_score=?,category=? WHERE id=?""",(edit_name.strip(),edit_id.strip(),edit_department,edit_semester,*[edited_values[key] for key in FEATURES],edited_score,edited_category,int(record_id)))
                        conn.commit()
                        st.success("Student profile updated and the estimate recalculated.")
                        st.rerun()
            with st.expander("🗑️ Delete a student record"):
                delete_id=st.selectbox("Select record ID to delete",ids,key="delete_record_id")
                confirm_delete=st.checkbox("I understand this permanently deletes this local prediction record.",key="confirm_record_delete")
                if st.button("Delete selected record",disabled=not confirm_delete):
                    conn.execute("DELETE FROM students WHERE id=?",(int(delete_id),)); conn.commit(); st.success("Record deleted."); st.rerun()
        st.caption("Records are stored locally in students.db on this computer.")

elif page=="Batch CSV Prediction":
    st.subheader("📦 Batch Prediction from CSV")
    st.write("Upload a CSV with the exact model feature columns. Optional columns (such as student name or ID) are retained in output.")
    st.code(", ".join(FEATURES))
    uploaded=st.file_uploader("Upload student feature CSV",type=["csv"])
    batch_models=list(bundle.get("models",{})) or ["Model unavailable"]
    model_name=st.selectbox("Model for batch prediction",batch_models,key="batchmodel",disabled=not bundle.get("models"))
    st.download_button("⬇️ Download model-feature CSV template",pd.DataFrame(columns=FEATURES).to_csv(index=False).encode("utf-8"),"student_input_template.csv","text/csv")
    if uploaded:
        try:
            batch=pd.read_csv(uploaded,encoding="utf-8-sig")
            batch.columns=[str(column).replace("\ufeff","").strip() for column in batch.columns]
            missing=[c for c in FEATURES if c not in batch.columns]
            if missing:
                st.error("Missing required columns: "+", ".join(missing))
                st.info("Rename or add the missing headers using the downloadable template. No rows were processed.")
            elif batch.empty:
                st.warning("The CSV contains no data rows.")
            else:
                st.markdown("#### Uploaded data preview")
                st.dataframe(batch.head(8),use_container_width=True,hide_index=True)
                numeric_batch=batch.copy()
                invalid_counts={}
                for feature in FEATURES:
                    converted=pd.to_numeric(numeric_batch[feature],errors="coerce")
                    invalid_counts[feature]=int((numeric_batch[feature].notna() & converted.isna()).sum())
                    numeric_batch[feature]=converted
                invalid_mask=numeric_batch[FEATURES].isna().any(axis=1)
                entirely_invalid=numeric_batch[FEATURES].isna().all(axis=1)
                if entirely_invalid.any():
                    st.error(f"{int(entirely_invalid.sum())} row(s) have no usable feature values. They will not be removed; the model's median imputer will estimate missing fields. Review these rows before interpreting results.")
                elif invalid_mask.any():
                    st.warning(f"{int(invalid_mask.sum())} row(s) contain missing or non-numeric features. Rows are retained and model median imputation is used; please review the flagged rows.")
                invalid_detail={feature:count for feature,count in invalid_counts.items() if count}
                if invalid_detail:
                    st.warning("Non-numeric entries converted to missing: "+", ".join(f"{feature}: {count}" for feature,count in invalid_detail.items()))
                duplicate_rows=int(numeric_batch.duplicated(subset=FEATURES,keep=False).sum())
                if duplicate_rows:
                    st.info(f"{duplicate_rows} duplicate feature row(s) found. All rows are retained in the predictions.")
                with st.spinner(f"Generating {len(numeric_batch):,} predictions..."):
                    st.progress(35,text="Validating features")
                    prediction_values=np.clip(bundle["models"][model_name].predict(numeric_batch[FEATURES]),0,100).round(1)
                    st.progress(100,text="Predictions ready")
                numeric_batch["predicted_score"]=prediction_values
                numeric_batch["performance_band"]=numeric_batch["predicted_score"].apply(category_for)
                numeric_batch["input_review"]=np.where(invalid_mask,"Review missing/invalid feature values","All model features provided")
                low_attendance=numeric_batch["attendance_percent"].notna() & (numeric_batch["attendance_percent"]<75)
                low_estimate=numeric_batch["predicted_score"]<50
                numeric_batch["support_flag"]=np.select([low_estimate,low_attendance,numeric_batch["attendance_percent"].isna()], ["Review support needs","Review support needs","Review attendance data"],default="No automatic flag")
                st.session_state["latest_batch_results"]=numeric_batch.copy()
                metric1,metric2,metric3=st.columns(3)
                metric1.metric("Rows processed",f"{len(numeric_batch):,}")
                metric2.metric("Average model estimate",f"{numeric_batch['predicted_score'].mean():.1f}/100")
                metric3.metric("Support review flags",f"{int((numeric_batch['support_flag']=='Review support needs').sum()):,}")
                st.dataframe(numeric_batch,use_container_width=True,hide_index=True)
                st.download_button("📥 Download complete batch results",numeric_batch.to_csv(index=False).encode("utf-8-sig"),"batch_predictions.csv","text/csv")
        except Exception as e: st.error(f"Could not process this CSV: {e}")

elif page=="At-Risk Alerts":
    st.subheader("Student Support Review")
    st.caption("These configurable alerts use saved model estimates and available attendance fields. They support educator review; they are not definitive judgments or actual results.")
    records=pd.read_sql_query("SELECT * FROM students ORDER BY id DESC",conn)
    if records.empty:
        st.markdown('<div class="empty-state">No saved prediction records are available for risk review. Add a prediction first; no students are inferred or fabricated.</div>',unsafe_allow_html=True)
    else:
        settings1,settings2,settings3=st.columns(3)
        high_threshold=settings1.slider("High-review score threshold",0.0,100.0,float(st.session_state.get("risk_high_threshold",50.0)),1.0,key="risk_high_threshold")
        moderate_threshold=settings2.slider("Moderate-review score threshold",high_threshold,100.0,float(max(high_threshold,st.session_state.get("risk_moderate_threshold",65.0))),1.0,key="risk_moderate_threshold")
        attendance_threshold=settings3.slider("Attendance review threshold (%)",0.0,100.0,float(st.session_state.get("risk_attendance_threshold",75.0)),1.0,key="risk_attendance_threshold")
        records["Risk level"]=records.apply(lambda row:risk_level_for(row["predicted_score"],row["attendance_percent"],high_threshold,moderate_threshold,attendance_threshold),axis=1)
        records["Review reason"]=records.apply(lambda row:"Predicted score below high-review threshold" if row["predicted_score"]<high_threshold else ("Predicted score below moderate-review threshold" if row["predicted_score"]<moderate_threshold else ("Attendance below configured threshold" if row["attendance_percent"]<attendance_threshold else "No configured alert condition")),axis=1)
        records["Suggested supportive action"]=records.apply(lambda row:advice_for(row["predicted_score"],row["attendance_percent"],row["study_hours_per_day"]),axis=1)
        high_count=int((records["Risk level"]=="High").sum())
        moderate_count=int((records["Risk level"]=="Moderate").sum())
        low_count=int((records["Risk level"]=="Low").sum())
        k1,k2,k3=st.columns(3)
        k1.metric("High review",f"{high_count:,}")
        k2.metric("Moderate review",f"{moderate_count:,}")
        k3.metric("Low review",f"{low_count:,}")
        risk_filter=st.multiselect("Show risk levels",["High","Moderate","Low"],default=["High","Moderate"],key="risk_filter")
        attention=records[records["Risk level"].isin(risk_filter)]
        display_columns=[column for column in ["student_name","student_id","department","semester","predicted_score","attendance_percent","Risk level","Review reason","Suggested supportive action","created_at"] if column in attention.columns]
        if attention.empty:
            st.success("No saved records match the selected review levels.")
        else:
            st.dataframe(attention[display_columns],use_container_width=True,hide_index=True)
        st.download_button("Download full risk review CSV",records.to_csv(index=False).encode("utf-8-sig"),"student_risk_review.csv","text/csv")
        st.download_button("Download risk review PDF",dataframe_pdf_report("Student Support Review",records,"Configurable review signals from saved model estimates; not a diagnosis."),"student_risk_review.pdf","application/pdf")

elif page=="Model Lab":
    st.subheader("🧪 Model Performance Comparison")
    model_file=bool(bundle.get("models"))
    metrics_file=METRICS_PATH.exists()
    status1,status2,status3=st.columns(3)
    status1.metric("Model artifact", "Available" if model_file else "Missing")
    status2.metric("Metrics artifact", "Available" if metrics_file else "Missing")
    status3.metric("Training rows",f"{len(data):,}")
    available_metrics=metrics.get("models",{})
    if not available_metrics:
        st.warning("No saved model evaluation metrics were found. Metrics are not generated or fabricated on app startup.")
    else:
        comparison=pd.DataFrame(available_metrics).T
        model_names=list(bundle.get("models",{}))
        score_columns=[metric for metric in ["MAE","RMSE","R2"] if metric in comparison.columns]
        for name in model_names:
            with st.container(border=True):
                st.markdown(f"#### {name}"+(" · selected" if name==metrics.get("selected_model") else ""))
                if name in comparison.index:
                    cols=st.columns(max(1,len(score_columns)))
                    for index,metric in enumerate(score_columns):
                        cols[index].metric(metric,comparison.loc[name,metric])
        st.success(f"Selected baseline model by saved held-out R² comparison: {metrics.get('selected_model','Not recorded')}")
        st.caption("MAE and RMSE measure average error magnitude (lower is better). R² measures variance explained and can be negative; higher is better. These values describe the saved synthetic held-out evaluation, not real-world performance.")
        if {"MAE","RMSE"}.issubset(comparison.columns):
            fig,ax=plt.subplots(); comparison[["MAE","RMSE"]].plot(kind="bar",ax=ax,color=["#6366f1","#06b6d4"]); ax.set_ylabel("Score error"); ax.set_xlabel("Algorithm"); ax.set_title("Held-out error comparison"); st.pyplot(fig); plt.close(fig)
        if "R2" in comparison.columns:
            fig,ax=plt.subplots(); comparison["R2"].plot(kind="bar",ax=ax,color="#8b5cf6"); ax.axhline(0,color="#475569",linewidth=1); ax.set_ylabel("R²"); ax.set_xlabel("Algorithm"); ax.set_title("Held-out R² comparison"); st.pyplot(fig); plt.close(fig)
        st.dataframe(comparison,use_container_width=True)
        st.download_button("Download model comparison CSV",comparison.to_csv().encode("utf-8-sig"),"model_comparison.csv","text/csv")
        st.download_button("Download metrics JSON",json.dumps(metrics,indent=2),"model_metrics.json","application/json")
    st.markdown("#### Training configuration")
    st.write(f"Algorithms: {', '.join(bundle.get('models',{}).keys()) or 'No model pipelines loaded'}")
    st.write(f"Features: {', '.join(bundle.get('features',FEATURES))}")
    st.write(f"Target: {TARGET} · Split: 80/20 · Random state: 42 · Dataset: {DATA_PATH.name}")
    if not model_file:
        st.error("The trained model file is missing. Run the existing training script manually only when you intend to regenerate model artifacts.")
    if not metrics_file:
        st.info("Model predictions are disabled until a compatible trained artifact and its metrics are restored.")

elif page=="Reports":
    st.subheader("Reports & Exports")
    st.caption(f"Create downloads from real saved or currently uploaded data. Generated {datetime.now():%Y-%m-%d %H:%M}.")
    student_records=pd.read_sql_query("SELECT * FROM students ORDER BY id DESC",conn)
    report_tabs=st.tabs(["Student report","Batch results","College results","Model metrics"])
    with report_tabs[0]:
        if student_records.empty:
            st.info("No saved student prediction records are available for a report.")
        else:
            selected_id=st.selectbox("Select saved student prediction",student_records["id"].tolist(),format_func=lambda item:f"#{item} · {student_records.loc[student_records['id']==item,'student_name'].iloc[0]}",key="report_student_id")
            report_row=student_records.loc[student_records["id"]==selected_id].iloc[0]
            report_inputs={feature:report_row[feature] for feature in FEATURES}
            st.write(f"Saved model estimate: **{report_row['predicted_score']:.1f}/100** · Band: **{report_row['category']}** · Saved: {report_row['created_at']}")
            st.download_button("Download selected student PDF",pdf_report(report_row["student_name"],report_row["student_id"],report_row["department"],report_row["semester"],report_inputs,float(report_row["predicted_score"]),"Model used was not recorded in the database"),file_name=f"student_report_{selected_id}.pdf",mime="application/pdf")
            st.download_button("Download saved student records CSV",student_records.to_csv(index=False).encode("utf-8-sig"),"student_records_report.csv","text/csv")
    with report_tabs[1]:
        batch_report=st.session_state.get("latest_batch_results")
        if batch_report is None:
            st.info("Run a batch CSV prediction during this session to populate this report.")
        else:
            st.dataframe(batch_report.head(10),use_container_width=True,hide_index=True)
            st.download_button("Download batch CSV report",batch_report.to_csv(index=False).encode("utf-8-sig"),"batch_predictions_report.csv","text/csv")
            st.download_button("Download batch PDF summary",dataframe_pdf_report("Batch Prediction Summary",batch_report,"Predicted estimates from the selected saved model; inspect input review flags."),"batch_predictions_report.pdf","application/pdf")
    with report_tabs[2]:
        college_report=st.session_state.get("latest_college_results")
        if college_report is None:
            st.info("Upload and analyze a college result sheet during this session to populate this report.")
        else:
            st.dataframe(college_report.head(10),use_container_width=True,hide_index=True)
            st.download_button("Download college result CSV",college_report.to_csv(index=False).encode("utf-8-sig"),"college_analysis_report.csv","text/csv")
            st.download_button("Download college result PDF",dataframe_pdf_report("College Result Report",college_report,"Descriptive uploaded results; verify official grading policy."),"college_analysis_report.pdf","application/pdf")
    with report_tabs[3]:
        st.json(metrics)
        st.download_button("Download model metrics JSON",json.dumps(metrics,indent=2),"model_metrics.json","application/json")

else:
    st.subheader("Settings & Methodology")
    st.markdown("#### Project configuration")
    config1,config2,config3=st.columns(3)
    config1.metric("Dataset",DATA_PATH.name if DATA_PATH.exists() else "Missing")
    config2.metric("Model file", "Available" if MODEL_PATH.exists() else "Missing")
    config3.metric("Student database", "Available" if DB_PATH.exists() else "Not created")
    st.markdown("""
    **Project:** Student Performance Intelligence · AI-Powered Academic Analytics & Prediction.

    **Problem definition:** estimate a final score from selected academic and study indicators.

    **Dataset collection:** the included `student_performance.csv` is synthetic demonstration data (500 records), not real student data.

    **Preprocessing:** feature/target separation, 80/20 train-test split, median imputation; feature scaling for Linear Regression.

    **Model development:** Linear Regression, Decision Tree Regressor and Random Forest Regressor.

    **Evaluation:** MAE, RMSE and R² on a held-out test set.

    **Optimization:** compare models and select the model with the highest test-set R² for the baseline. Further cross-validation and hyperparameter search are future improvements.

    **Result analysis:** dashboard plots, college semester analysis, model metrics, predictions, CSV batch output, configurable support alerts and PDF reports.

    **Metric interpretation:** MAE/RMSE are error magnitudes; R² may be negative and is not a percentage accuracy.

    **Data checks:** prediction feature fields are defined by the saved pipeline. Missing batch features are median-imputed by the fitted pipeline and flagged in output. College grade policy is not inferred where it is absent.

    **Ethics and limits:** a prediction is only an estimate. Use permissioned, representative data, protect student privacy, evaluate fairness, and involve educators before real-world use. The local SQLite database is not encrypted; avoid sensitive real student information.
    """)
    st.subheader("🔎 Dataset preview")
    st.dataframe(data.head(15),use_container_width=True,hide_index=True)
    st.download_button("Download source dataset CSV",data.to_csv(index=False).encode("utf-8"),"student_performance.csv","text/csv")

st.markdown('<div class="app-footer">Student Performance Intelligence · Streamlit · pandas · scikit-learn · SQLite · ReportLab<br>Educational decision support only · Protect student data and follow institutional grading policy.</div>',unsafe_allow_html=True)
