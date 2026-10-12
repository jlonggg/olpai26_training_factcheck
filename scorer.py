#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trình chấm Bài 1 (Kiểm chứng tin tức) chạy trên máy của thí sinh.

Hàm score() tính macro-F1 trên ba nhãn SUPPORTED, REFUTED, NEI, giống hệt
công thức dùng khi chấm bài nộp.

Cách dùng:
    python3 scorer.py du_doan.csv --gt dap_an.csv
    python3 scorer.py du_doan.csv --gt dap_an.csv --show 10     # in thêm 10 nhận định bị đoán sai

Tệp đáp án cần hai cột id, label (ví dụ phần validation tự tách từ train.csv).
Nếu tệp đáp án có thêm cột claim thì --show in cả nội dung nhận định.
Tệp dự đoán cần hai cột id, label. Dòng thiếu hoặc nhãn lạ được tính là dự đoán sai.
"""
import argparse, sys
import pandas as pd


def _labels(series):
    return series.fillna("").astype(str).str.strip().str.upper()


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    try:
        key = row_id_column_name
        sol = pd.DataFrame({key: solution[key].astype(str).str.strip(), "gold": _labels(solution["label"])})
        if key not in submission.columns or "label" not in submission.columns:
            return 0.0
        sub = pd.DataFrame({key: submission[key].astype(str).str.strip(), "pred": _labels(submission["label"])})
        sub = sub.drop_duplicates(key, keep="first")
        m = sol.merge(sub, on=key, how="left")
        pred, gold = m["pred"].fillna(""), m["gold"]
        f1 = []
        for c in ("SUPPORTED", "REFUTED", "NEI"):
            tp = int(((pred == c) & (gold == c)).sum())
            fp = int(((pred == c) & (gold != c)).sum())
            fn = int(((pred != c) & (gold == c)).sum())
            f1.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
        return float(sum(f1) / len(f1))
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser(description="Chấm Bài 1 bằng macro-F1")
    ap.add_argument("pred"); ap.add_argument("--gt", required=True); ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    gt, pred = pd.read_csv(a.gt, dtype=str), pd.read_csv(a.pred, dtype=str)
    if "id" not in pred.columns or "label" not in pred.columns:
        sys.exit(f"Tệp dự đoán phải có hai cột id, label; các cột đang có là {list(pred.columns)}")
    s = score(gt, pred, "id")
    m = gt.merge(pred.drop_duplicates("id"), on="id", how="left", suffixes=("", "_pred"))
    g, p = _labels(m["label"]), _labels(m["label_pred"])
    missing = int(m["label_pred"].isna().sum())
    print(f"macro-F1 = {s:.4f}   (accuracy = {(g == p).mean():.4f}, số dòng chấm = {len(m)}, dòng thiếu = {missing})")
    print(f"{'nhãn':<10} {'precision':>9} {'recall':>7} {'F1':>7} {'số mẫu':>7}")
    for c in ("SUPPORTED", "REFUTED", "NEI"):
        tp, fp, fn = int(((p == c) & (g == c)).sum()), int(((p == c) & (g != c)).sum()), int(((p != c) & (g == c)).sum())
        pr, rc = (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)
        print(f"{c:<10} {pr:>9.3f} {rc:>7.3f} {(2 * pr * rc / (pr + rc) if pr + rc else 0.0):>7.3f} {int((g == c).sum()):>7}")
    print("\nMa trận nhầm lẫn (hàng là nhãn đúng, cột là nhãn dự đoán):")
    print(pd.crosstab(g.rename("đúng"), p.rename("dự đoán")).to_string())
    if a.show and "claim" in m.columns:
        print(f"\n{a.show} nhận định bị đoán sai đầu tiên:")
        for _, r in m[g != p].head(a.show).iterrows():
            print(f"- [{r['label']} bị đoán là {r['label_pred']}] {r['claim']}")


if __name__ == "__main__":
    main()
