import os
import json
from typing import Dict, List, Optional, Tuple


BASELINE_RESULT_DIR = r"d:\Python File\graduate_project\data\baseline_result"


def to_label(value) -> Optional[int]:
    """将标签转成 0/1，失败返回 None。"""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value if value in (0, 1) else None
    if isinstance(value, float):
        int_v = int(value)
        return int_v if int_v in (0, 1) else None
    if isinstance(value, str):
        text = value.strip()
        if text in {"0", "1"}:
            return int(text)
    return None


def compute_confusion(rows: List[Tuple[int, int]]) -> Dict[str, int]:
    """
    rows: List[(truth, pred)]
    正类定义为 1。
    """
    tp = tn = fp = fn = 0
    for truth, pred in rows:
        if truth == 1 and pred == 1:
            tp += 1
        elif truth == 0 and pred == 0:
            tn += 1
        elif truth == 0 and pred == 1:
            fp += 1
        elif truth == 1 and pred == 0:
            fn += 1
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator != 0 else 0.0


def compute_metrics(rows: List[Tuple[int, int]]) -> Dict[str, float]:
    cm = compute_confusion(rows)
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    total = tp + tn + fp + fn

    accuracy = safe_div(tp + tn, total)
    recall = safe_div(tp, tp + fn)
    precision = safe_div(tp, tp + fp)
    f1 = safe_div(2 * precision * recall, precision + recall)
    specificity = safe_div(tn, tn + fp)
    fpr = safe_div(fp, fp + tn)

    return {
        "total": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1_score": f1,
        "specificity": specificity,
        "false_positive_rate": fpr,
    }


def load_eval_rows_from_jd_folder(jd_folder_path: str) -> Tuple[List[Tuple[int, int]], int]:
    """读取某个 JD 文件夹内所有 *_eval.json，返回有效样本与无效样本数。"""
    rows: List[Tuple[int, int]] = []
    invalid_count = 0

    for name in os.listdir(jd_folder_path):
        if not name.endswith(".json"):
            continue
        file_path = os.path.join(jd_folder_path, name)
        if not os.path.isfile(file_path):
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            invalid_count += 1
            continue

        truth = to_label(obj.get("ground_truth"))
        pred = to_label(obj.get("result"))

        if truth is None or pred is None:
            invalid_count += 1
            continue

        rows.append((truth, pred))

    return rows, invalid_count


def format_ratio(x: float) -> str:
    return f"{x:.4f} ({x * 100:.2f}%)"


def print_metrics_block(title: str, metrics: Dict[str, float], invalid_count: int) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"有效样本数: {int(metrics['total'])}")
    print(f"无效样本数: {invalid_count}")
    print(f"TP: {int(metrics['tp'])}, TN: {int(metrics['tn'])}, FP: {int(metrics['fp'])}, FN: {int(metrics['fn'])}")
    print(f"Accuracy      : {format_ratio(metrics['accuracy'])}")
    print(f"Recall        : {format_ratio(metrics['recall'])}")
    print(f"Precision     : {format_ratio(metrics['precision'])}")
    print(f"F1-score      : {format_ratio(metrics['f1_score'])}")
    print(f"Specificity   : {format_ratio(metrics['specificity'])}")
    print(f"False Pos Rate: {format_ratio(metrics['false_positive_rate'])}")


def main() -> None:
    if not os.path.exists(BASELINE_RESULT_DIR):
        print(f"结果目录不存在: {BASELINE_RESULT_DIR}")
        return

    jd_dirs = [
        d for d in os.listdir(BASELINE_RESULT_DIR)
        if os.path.isdir(os.path.join(BASELINE_RESULT_DIR, d))
    ]
    jd_dirs.sort()

    if not jd_dirs:
        print(f"目录中未找到按 JD 划分的结果子目录: {BASELINE_RESULT_DIR}")
        return

    all_rows: List[Tuple[int, int]] = []
    total_invalid = 0
    summary: Dict[str, Dict[str, float]] = {}

    for jd in jd_dirs:
        jd_path = os.path.join(BASELINE_RESULT_DIR, jd)
        rows, invalid_count = load_eval_rows_from_jd_folder(jd_path)
        metrics = compute_metrics(rows)

        all_rows.extend(rows)
        total_invalid += invalid_count

        summary[jd] = {
            **metrics,
            "invalid_count": invalid_count,
        }

        print_metrics_block(f"JD: {jd}", metrics, invalid_count)

    overall_metrics = compute_metrics(all_rows)
    print_metrics_block("Overall", overall_metrics, total_invalid)

    output = {
        "by_jd": summary,
        "overall": {
            **overall_metrics,
            "invalid_count": total_invalid,
        },
    }

    output_path = os.path.join(os.path.dirname(__file__), "baseline_metrics_summary.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n指标汇总已保存:")
    print(output_path)


if __name__ == "__main__":
    main()
