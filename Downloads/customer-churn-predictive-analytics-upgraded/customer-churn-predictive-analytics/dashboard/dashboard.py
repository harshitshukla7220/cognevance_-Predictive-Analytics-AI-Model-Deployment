from pathlib import Path
import json
import pandas as pd
import streamlit as st
from src.inference import load_bundle, predict_records, MODEL_PATH

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="Customer Churn Analytics", page_icon="📊", layout="wide")
st.title("📊 Customer Churn Prediction & Analytics")
st.caption("End-to-end predictive analytics: data quality, model evaluation and customer-level predictions.")

metrics_path = ROOT/"reports/model_metrics.json"
if not MODEL_PATH.exists() or not metrics_path.exists():
    st.warning("Model or reports not found. Run `python -m src.train` from the project root first.")
    st.stop()
bundle = load_bundle()
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
data = pd.read_csv(ROOT/"data/customer_churn.csv")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Customers", f"{len(data):,}")
c2.metric("Churn rate", f"{(data.Churn=='Yes').mean()*100:.1f}%")
c3.metric("Best model", bundle["model_name"])
c4.metric("Test ROC-AUC", f"{metrics['final_metrics']['roc_auc']:.3f}")
tab1,tab2,tab3 = st.tabs(["Overview & model comparison","Predict customer","Reports"])
with tab1:
    st.subheader("Customer churn distribution")
    st.bar_chart(data["Churn"].value_counts())
    st.subheader("Model comparison")
    comp = pd.DataFrame(metrics["models"]).set_index("model")
    st.dataframe(comp[["cv_pr_auc_mean","accuracy","precision","recall","f1","roc_auc","pr_auc"]].style.format("{:.3f}"), use_container_width=True)
    st.bar_chart(comp[["accuracy","f1","roc_auc","pr_auc"]])
    st.image(str(ROOT/"images/confusion_matrix.png"), caption="Best model confusion matrix")
with tab2:
    st.subheader("Enter customer information")
    schema = data.drop(columns=["customerID","Churn"])
    row = {}
    left,right = st.columns(2)
    for i,col in enumerate(schema.columns):
        target = left if i%2==0 else right
        series = schema[col]
        if col == "TotalCharges":
            row[col] = target.number_input(col, min_value=0.0, max_value=100000.0, value=1000.0)
        elif pd.api.types.is_numeric_dtype(series):
            lo=float(series.min()); hi=float(series.max()); default=float(series.median())
            row[col] = target.number_input(col, min_value=lo, max_value=hi, value=default)
        else:
            opts=sorted(series.dropna().astype(str).unique().tolist())
            row[col] = target.selectbox(col, opts, index=0)
    if st.button("Predict churn", type="primary"):
        try:
            result = predict_records(bundle,[row])[0]
            st.metric("Churn probability", f"{result['churn_probability']*100:.1f}%")
            st.write("Prediction:", result["prediction"])
            st.progress(result["churn_probability"])
            st.json(result)
        except Exception as exc:
            st.error(str(exc))
with tab3:
    st.subheader("Evaluation report")
    st.json(metrics["final_metrics"])
    st.download_button("Download model metrics JSON", metrics_path.read_bytes(), file_name="model_metrics.json")
    report_path=ROOT/"reports/model_report.md"
    st.download_button("Download model report", report_path.read_bytes(), file_name="model_report.md")
