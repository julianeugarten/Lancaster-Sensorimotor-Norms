# %%

import json
from pathlib import Path

ts = time.strftime("%Y-%m-%d")
print(f"Timestamp: {ts}")

CWD = Path(__file__).parent
OUT_DIR = CWD / "OUT_DIR"

# %%

def json_to_perclass_latex(json_path, caption, label):
    with open(json_path) as f:
        r = json.load(f)

    names = r["class_display_names"]
    per_class = r["per_class_metrics"]
    overall = r["overall_metrics"]

    def cell(name, metric):
        d = per_class[name][metric]
        return f"{d['mean']:.4f} $\\pm$ {d['std']:.3f}"

    def macro_cell(metric_key):
        d = overall[metric_key]
        return f"{d['mean']:.4f} $\\pm$ {d['std']:.3f}"

    col_spec = "c" * len(names)
    header_names = " & ".join(f"\\textbf{{{n}}}" for n in names)

    lines = []
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(f"\\begin{{tabular}}{{l{col_spec}|lc}}")
    lines.append(r"\toprule")
    lines.append(f" & \\multicolumn{{{len(names)}}}{{c}}{{\\textbf{{Per class}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{Overall}}}}\\\\")
    lines.append(f"\\cmidrule{{2-{len(names)+2}}}")
    lines.append(f"\\textbf{{Metric}} & {header_names} & \\textbf{{Metric}} & \\textbf{{Macros}}\\\\")
    lines.append(r"\midrule")

    metric_labels = {"precision": "Precision", "recall": "Recall", "f1": "F1"}
    macro_keys = {"precision": "macro_precision", "recall": "macro_recall", "f1": "macro_f1"}

    for metric, label_str in metric_labels.items():
        row_cells = " & ".join(cell(n, metric) for n in names)
        macro = macro_cell(macro_keys[metric])
        lines.append(f"{label_str} & {row_cells} & Macro {label_str} & {macro}\\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    auc = overall["roc_auc_ovr"]
    lines.append(f"\\caption{{{caption} Macro ROC AUC (one-vs-rest) $= {auc['mean']:.3f} \\pm {auc['std']:.3f}$.}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)

# %%

for mode, cap in [
    ("three_class", "three-regime classifier"),
    ("three_class_mfw6", "6-word MFW baseline"),
    ("three_class_mfw100", "100-word MFW baseline"),
]:
    tex = json_to_perclass_latex(
        OUT_DIR / f"{ts}_{mode}_classification_results.json",
        caption=f"\\textbf{{Per-Class Performance}} across 5 folds (mean $\\pm$ std), {cap}.",
        label=f"tab:perclass_{mode}",
    )
    print(tex, "\n\n")