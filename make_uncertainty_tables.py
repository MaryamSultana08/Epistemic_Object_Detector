import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create paper-ready quantitative uncertainty tables from uncertainty_eval output."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing detection_uncertainty.csv and summary.json",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for tables (default: <input_dir>/paper_tables)",
    )
    parser.add_argument(
        "--round_digits",
        type=int,
        default=4,
        help="Decimal places for saved table values",
    )
    parser.add_argument(
        "--topk_class_latex",
        type=int,
        default=20,
        help="Number of classes to include in the LaTeX class table (CSV keeps all classes).",
    )
    return parser.parse_args()


def read_inputs(input_dir: Path):
    det_path = input_dir / "detection_uncertainty.csv"
    summary_path = input_dir / "summary.json"

    if not det_path.exists():
        raise FileNotFoundError(f"Missing file: {det_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing file: {summary_path}")

    det = pd.read_csv(det_path)
    with summary_path.open() as f:
        summary = json.load(f)
    return det, summary


def safe_stats(series: pd.Series):
    if len(series) == 0:
        return dict(mean=np.nan, std=np.nan, p50=np.nan, p90=np.nan, p95=np.nan)
    return dict(
        mean=float(series.mean()),
        std=float(series.std(ddof=0)),
        p50=float(series.quantile(0.50)),
        p90=float(series.quantile(0.90)),
        p95=float(series.quantile(0.95)),
    )


def build_overall_table(summary: dict, det: pd.DataFrame):
    fp_ratio = float((det["is_tp"] == 0).mean()) if len(det) else np.nan
    tp_ratio = float((det["is_tp"] == 1).mean()) if len(det) else np.nan

    row = {
        "images_processed": summary.get("images_processed", np.nan),
        "num_detections": summary.get("num_detections", len(det)),
        "total_epistemic_mean": summary.get("total_epistemic_mean", np.nan),
        "total_epistemic_std": summary.get("total_epistemic_std", np.nan),
        "total_epistemic_p50": summary.get("total_epistemic_p50", np.nan),
        "total_epistemic_p75": summary.get("total_epistemic_p75", np.nan),
        "total_epistemic_p90": summary.get("total_epistemic_p90", np.nan),
        "total_epistemic_p95": summary.get("total_epistemic_p95", np.nan),
        "tp_count": summary.get("tp_count", int((det["is_tp"] == 1).sum())),
        "fp_count": summary.get("fp_count", int((det["is_tp"] == 0).sum())),
        "tp_ratio": tp_ratio,
        "fp_ratio": fp_ratio,
        "tp_total_epistemic_mean": summary.get("tp_total_epistemic_mean", np.nan),
        "fp_total_epistemic_mean": summary.get("fp_total_epistemic_mean", np.nan),
        "fp_minus_tp_total_epistemic": summary.get("fp_minus_tp_total_epistemic", np.nan),
    }
    return pd.DataFrame([row])


def build_outcome_table(det: pd.DataFrame):
    rows = []
    for outcome, flag in [("TP", 1), ("FP", 0)]:
        sub = det.loc[det["is_tp"] == flag, "total_epistemic"]
        s = safe_stats(sub)
        rows.append(
            {
                "outcome": outcome,
                "count": int((det["is_tp"] == flag).sum()),
                "mean_total_epistemic": s["mean"],
                "std_total_epistemic": s["std"],
                "p50_total_epistemic": s["p50"],
                "p90_total_epistemic": s["p90"],
                "p95_total_epistemic": s["p95"],
            }
        )
    return pd.DataFrame(rows)


def build_class_table(det: pd.DataFrame):
    g = det.groupby(["label", "label_name"], as_index=False).agg(
        num_detections=("image_id", "count"),
        mean_score=("score", "mean"),
        mean_total_epistemic=("total_epistemic", "mean"),
        mean_reg_epistemic=("reg_epistemic", "mean"),
        mean_cls_epistemic=("cls_epistemic", "mean"),
        tp_count=("is_tp", lambda x: int((x == 1).sum())),
        fp_count=("is_tp", lambda x: int((x == 0).sum())),
    )
    g["fp_rate"] = g["fp_count"] / g["num_detections"].clip(lower=1)
    g = g.sort_values(["mean_total_epistemic", "num_detections"], ascending=[False, False])
    return g


def build_uncertainty_bin_table(det: pd.DataFrame):
    bins = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.01]
    labels = ["0.00-0.05", "0.05-0.10", "0.10-0.15", "0.15-0.20", "0.20-0.30", "0.30-0.50", "0.50-1.00"]
    b = det.copy()
    b["u_bin"] = pd.cut(b["total_epistemic"], bins=bins, labels=labels, include_lowest=True, right=False)
    out = b.groupby("u_bin", as_index=False, observed=False).agg(
        num_detections=("image_id", "count"),
        tp_count=("is_tp", lambda x: int((x == 1).sum())),
        fp_count=("is_tp", lambda x: int((x == 0).sum())),
        mean_score=("score", "mean"),
        mean_total_epistemic=("total_epistemic", "mean"),
    )
    out["fp_rate"] = out["fp_count"] / out["num_detections"].clip(lower=1)
    return out


def build_uncertainty_quantile_table(det: pd.DataFrame, n_quantiles: int = 10):
    q = det.copy()
    q["u_quantile"] = pd.qcut(q["total_epistemic"], q=n_quantiles, labels=False, duplicates="drop")
    out = q.groupby("u_quantile", as_index=False).agg(
        num_detections=("image_id", "count"),
        tp_count=("is_tp", lambda x: int((x == 1).sum())),
        fp_count=("is_tp", lambda x: int((x == 0).sum())),
        mean_score=("score", "mean"),
        mean_total_epistemic=("total_epistemic", "mean"),
        min_total_epistemic=("total_epistemic", "min"),
        max_total_epistemic=("total_epistemic", "max"),
    )
    out["fp_rate"] = out["fp_count"] / out["num_detections"].clip(lower=1)
    return out


def build_markdown_report(
    overall: pd.DataFrame,
    outcome: pd.DataFrame,
    ubin: pd.DataFrame,
    out_path: Path,
):
    with out_path.open("w") as f:
        f.write("# Epistemic Uncertainty Tables\n\n")
        f.write("## Overall\n\n")
        f.write(overall.to_markdown(index=False))
        f.write("\n\n## TP vs FP\n\n")
        f.write(outcome.to_markdown(index=False))
        f.write("\n\n## Uncertainty Bins\n\n")
        f.write(ubin.to_markdown(index=False))
        f.write("\n")


def write_latex_table(
    df: pd.DataFrame,
    out_path: Path,
    caption: str,
    label: str,
    digits: int,
):
    latex = df.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
        float_format=lambda x: f"{x:.{int(digits)}f}",
        na_rep="",
    )
    with out_path.open("w") as f:
        f.write(latex)


def build_latex_bundle(
    out_path: Path,
    latex_paths: dict,
):
    with out_path.open("w") as f:
        f.write("% Auto-generated LaTeX table include file.\n")
        f.write("% Requires \\usepackage{booktabs} in your main paper.\n\n")
        f.write("% Overall uncertainty table\n")
        f.write(f"\\input{{{latex_paths['overall']}}}\n\n")
        f.write("% TP vs FP table\n")
        f.write(f"\\input{{{latex_paths['tp_fp']}}}\n\n")
        f.write("% Uncertainty bin table\n")
        f.write(f"\\input{{{latex_paths['u_bins']}}}\n\n")
        f.write("% Top-K class table\n")
        f.write(f"\\input{{{latex_paths['class_topk']}}}\n")


def round_numeric(df: pd.DataFrame, digits: int):
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].round(int(digits))
    return out


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else (input_dir / "paper_tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    det, summary = read_inputs(input_dir)

    overall = build_overall_table(summary, det)
    outcome = build_outcome_table(det)
    by_class = build_class_table(det)
    u_bin = build_uncertainty_bin_table(det)
    u_quant = build_uncertainty_quantile_table(det)

    overall_r = round_numeric(overall, args.round_digits)
    outcome_r = round_numeric(outcome, args.round_digits)
    by_class_r = round_numeric(by_class, args.round_digits)
    u_bin_r = round_numeric(u_bin, args.round_digits)
    u_quant_r = round_numeric(u_quant, args.round_digits)

    overall_path = output_dir / "table_overall.csv"
    outcome_path = output_dir / "table_tp_fp.csv"
    class_path = output_dir / "table_by_class.csv"
    ubin_path = output_dir / "table_uncertainty_bins.csv"
    uquant_path = output_dir / "table_uncertainty_quantiles.csv"
    md_path = output_dir / "report_tables.md"
    tex_overall_path = output_dir / "table_overall.tex"
    tex_outcome_path = output_dir / "table_tp_fp.tex"
    tex_ubin_path = output_dir / "table_uncertainty_bins.tex"
    tex_class_topk_path = output_dir / "table_by_class_topk.tex"
    tex_bundle_path = output_dir / "tables_for_paper.tex"

    overall_r.to_csv(overall_path, index=False)
    outcome_r.to_csv(outcome_path, index=False)
    by_class_r.to_csv(class_path, index=False)
    u_bin_r.to_csv(ubin_path, index=False)
    u_quant_r.to_csv(uquant_path, index=False)
    build_markdown_report(overall_r, outcome_r, u_bin_r, md_path)

    class_topk = by_class_r.head(max(1, int(args.topk_class_latex)))

    write_latex_table(
        overall_r,
        tex_overall_path,
        caption="Overall epistemic uncertainty summary.",
        label="tab:uncert_overall",
        digits=args.round_digits,
    )
    write_latex_table(
        outcome_r,
        tex_outcome_path,
        caption="Epistemic uncertainty for true positives vs false positives.",
        label="tab:uncert_tp_fp",
        digits=args.round_digits,
    )
    write_latex_table(
        u_bin_r,
        tex_ubin_path,
        caption="Detection outcomes across epistemic uncertainty bins.",
        label="tab:uncert_bins",
        digits=args.round_digits,
    )
    write_latex_table(
        class_topk,
        tex_class_topk_path,
        caption=f"Top-{len(class_topk)} classes by mean epistemic uncertainty.",
        label="tab:uncert_class_topk",
        digits=args.round_digits,
    )
    build_latex_bundle(
        tex_bundle_path,
        latex_paths={
            "overall": tex_overall_path.name,
            "tp_fp": tex_outcome_path.name,
            "u_bins": tex_ubin_path.name,
            "class_topk": tex_class_topk_path.name,
        },
    )

    print("Wrote paper tables:")
    print(f"- {overall_path}")
    print(f"- {outcome_path}")
    print(f"- {class_path}")
    print(f"- {ubin_path}")
    print(f"- {uquant_path}")
    print(f"- {md_path}")
    print("Wrote LaTeX tables:")
    print(f"- {tex_overall_path}")
    print(f"- {tex_outcome_path}")
    print(f"- {tex_ubin_path}")
    print(f"- {tex_class_topk_path}")
    print(f"- {tex_bundle_path}")


if __name__ == "__main__":
    main()
