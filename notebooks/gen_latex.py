# %%

import json
from pathlib import Path
import time

ts = time.strftime("%Y-%m-%d")
print(f"Timestamp: {ts}")

CWD = Path(__file__).parent
OUT_DIR = CWD / "OUT_DIR"

# %%

def json_to_perclass_latex_rows(json_path, caption, label):
    with open(json_path) as f:
        r = json.load(f)

    names = r["class_display_names"]
    per_class = r["per_class_metrics"]
    overall = r["overall_metrics"]

    def cell(name, metric):
        d = per_class[name][metric]
        return f"{d['mean']:.3f} $\\pm$ {d['std']:.2f}"

    def macro_cell(metric_key):
        d = overall[metric_key]
        return f"{d['mean']:.3f} $\\pm$ {d['std']:.2f}"

    lines = []
    lines.append(r"\begin{table}[]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Class} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} \\")
    lines.append(r"\midrule")

    for name in names:
        row = f"{name} & {cell(name, 'precision')} & {cell(name, 'recall')} & {cell(name, 'f1')} \\\\"
        lines.append(row)

    lines.append(r"\midrule")
    macro_row = f"\\textbf{{Macro avg.}} & {macro_cell('macro_precision')} & {macro_cell('macro_recall')} & {macro_cell('macro_f1')} \\\\"
    lines.append(macro_row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    auc = overall["roc_auc_ovr"]
    acc = overall["accuracy"]
    lines.append(f"\\caption{{{caption} Accuracy $= {acc['mean']:.3f} \\pm {acc['std']:.3f}$; Macro ROC AUC (one-vs-rest) $= {auc['mean']:.3f} \\pm {auc['std']:.3f}$.}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\end{table}")

    return "\n".join(lines)

# %%

for mode, cap in [
    ("three_class", "3-class classifier"),
    ("three_class_mfw6", "6-word MFW baseline"),
    ("three_class_mfw100", "100-word MFW baseline"),
]:
    tex = json_to_perclass_latex_rows(
        OUT_DIR / f"{ts}_{mode}_classification_results.json",
        caption=f"\\textbf{{Per-Class Performance}} across 5 folds (mean $\\pm$ std), {cap}.",
        label=f"tab:perclass_{mode}",
    )
    print(tex, "\n\n")
# %%

for mode, cap in [
    ("four_class", "4-class classifier"),
    ("four_class_mfw6", "6-word MFW baseline"),
    ("four_class_mfw100", "100-word MFW baseline"),
]:
    tex = json_to_perclass_latex_rows(
        OUT_DIR / f"{ts}_{mode}_classification_results.json",
        caption=f"\\textbf{{Per-Class Performance}} across 5 folds (mean $\\pm$ std), {cap}.",
        label=f"tab:perclass_{mode}",
    )
    print(tex, "\n\n")


# %%
