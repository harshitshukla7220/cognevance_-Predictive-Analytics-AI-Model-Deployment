import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT/"data/customer_churn.csv")
report = {
    "rows": int(len(df)), "columns": int(df.shape[1]),
    "missing_values": {k:int(v) for k,v in df.isna().sum().items() if v},
    "duplicate_rows": int(df.duplicated().sum()),
    "target_distribution": {str(k):int(v) for k,v in df["Churn"].value_counts().items()},
    "target_rate": float((df["Churn"]=="Yes").mean()),
    "numeric_summary": df.select_dtypes(include="number").describe().to_dict()
}
(ROOT/"reports").mkdir(exist_ok=True)
(ROOT/"reports/data_quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({k:v for k,v in report.items() if k!="numeric_summary"}, indent=2))
