from pathlib import Path
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

REF = Path("data/processed/housing_processed.csv")
CUR = Path("data/monitoring/current_batch.csv")

ref = pd.read_csv(REF)
cur = pd.read_csv(CUR)

report = Report(
    metrics = [
        DataDriftPreset(),
        TargetDriftPreset(),
    ]
)

report.run(
    reference_data = ref,
    current_data = cur,
)

report.save_html("monitoring/evidently/drift_report.html")
print("Report Saved!")