"""Turn Ultralytics validation results into a simple table for the report."""
from datetime import datetime

TARGET_MAP50 = 0.90


def per_class_rows(metrics, names):
    """Returns [(name, precision, recall, ap50)] for every class that appears."""
    box = metrics.box
    rows = []
    for i, cid in enumerate(box.ap_class_index):
        p, r, ap50, _ = box.class_result(i)
        rows.append((names[int(cid)], float(p), float(r), float(ap50)))
    return rows


def summary_markdown(title, metrics, names, extra=None):
    rows = per_class_rows(metrics, names)
    map50 = float(metrics.box.map50)
    lines = [f"# {title}", "", f"Date: {datetime.now():%Y-%m-%d %H:%M}", ""]
    for k, v in (extra or {}).items():
        lines.append(f"- {k}: {v}")
    lines += ["", "| Product | Precision | Recall | AP@0.5 |", "| --- | --- | --- | --- |"]
    for name, p, r, ap in rows:
        flag = "" if ap >= TARGET_MAP50 else " (below 0.90)"
        lines.append(f"| {name} | {p:.3f} | {r:.3f} | {ap:.3f}{flag} |")
    verdict = "MEETS" if map50 >= TARGET_MAP50 else "DOES NOT MEET"
    lines += [f"| **All** | {float(metrics.box.mp):.3f} | {float(metrics.box.mr):.3f} | **{map50:.3f}** |",
              "", f"mAP@0.5 = {map50:.3f}: {verdict} the target of {TARGET_MAP50:.2f}."]
    weak = [n for n, _, _, ap in rows if ap < TARGET_MAP50]
    if weak:
        lines.append(f"Take more photos of: {', '.join(weak)} (especially in the light and angles it misses).")
    return "\n".join(lines) + "\n", map50
