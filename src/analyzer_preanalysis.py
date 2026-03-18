"""
analyzer_preanalysis.py
========================
Module 1.1 — Pre-Analysis: Data Quality & Audit Assessment

Câu hỏi trả lời:
  - Dữ liệu có đáng tin cậy không?
  - Mỗi quý báo cáo đã được kiểm toán chưa?
  - Tỷ lệ dữ liệu thiếu ảnh hưởng thế nào?
  - Chính sách kế toán chính (khấu hao, ghi nhận doanh thu) là gì?

Input:
  - audit_dict: dict of audit status Series (from data_loader.load_raw_with_audit)
  - data_dict: cleaned financial data
  - Thuyết minh sheet: scan for accounting policy keywords

Output: output/preanalysis_report.json
"""

import json
import os
import numpy as np
import pandas as pd


def _max_consecutive_streak(values, target='Chưa kiểm toán'):
    """Find the longest consecutive run of `target` in a list of values."""
    max_streak = 0
    current = 0
    for v in values:
        if str(v).strip() == target:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def _compute_missing_rate(data_dict):
    """Calculate overall NaN rate across BCĐKT, KQKD, LCTT numeric columns."""
    total_cells = 0
    total_missing = 0
    for sheet_name in ['Bảng cân đối kế toán', 'Kết quả kinh doanh', 'Lưu chuyển tiền tệ']:
        df = data_dict.get(sheet_name)
        if df is None:
            continue
        numeric_part = df.iloc[:, 1:]
        total_cells += numeric_part.size
        total_missing += numeric_part.isnull().sum().sum()
    if total_cells == 0:
        return 0.0
    return round(total_missing / total_cells * 100, 2)


def _extract_policy_keywords(data_dict):
    """Scan Thuyết minh for accounting policy keywords."""
    thuyet_minh = data_dict.get('Thuyết minh')
    if thuyet_minh is None:
        return []

    keywords = ['khấu hao', 'phương pháp', 'ghi nhận doanh thu', 'dự phòng',
                'chính sách kế toán', 'nguyên tắc', 'ước tính']
    pattern = '|'.join(keywords)

    col0 = thuyet_minh.iloc[:, 0].astype(str)
    mask = col0.str.contains(pattern, case=False, na=False, regex=True)
    found_rows = thuyet_minh.loc[mask, thuyet_minh.columns[0]].tolist()
    return found_rows


def run_preanalysis(data_dict, audit_dict, output_dir):
    """
    Entry point for Module 1.1 Pre-Analysis.
    Produces output/preanalysis_report.json
    """
    print("\n--- MODULE 1.1: PRE-ANALYSIS (DATA QUALITY) ---")

    # --- Bước 1 & 2: Audit status analysis ---
    audit_quarters = {}
    all_statuses = []

    for sheet_name in ['Bảng cân đối kế toán', 'Kết quả kinh doanh', 'Lưu chuyển tiền tệ']:
        series = audit_dict.get(sheet_name)
        if series is not None:
            for q_label, status in series.items():
                q_str = str(q_label).strip()
                s_str = str(status).strip()
                if q_str not in audit_quarters:
                    audit_quarters[q_str] = s_str
                all_statuses.append(s_str)

    total_quarters = len(audit_quarters)
    audited_quarters = sum(1 for s in audit_quarters.values() if s == 'Đã kiểm toán')
    audit_rate = round(audited_quarters / total_quarters * 100, 2) if total_quarters > 0 else 0.0

    # Max consecutive unaudited (use BCĐKT as primary — most complete)
    primary_audit = audit_dict.get('Bảng cân đối kế toán')
    if primary_audit is not None:
        max_consec = _max_consecutive_streak(primary_audit.values, 'Chưa kiểm toán')
    else:
        max_consec = 0

    # --- Bước 3: Missing rate ---
    missing_rate = _compute_missing_rate(data_dict)

    # --- Bước 4: Policy keywords from Thuyết minh ---
    policy_rows = _extract_policy_keywords(data_dict)

    # --- Data quality flag ---
    if audit_rate >= 50:
        flag = "OK"
    elif audit_rate >= 20:
        flag = "CAUTION"
    else:
        flag = "WARNING"

    report = {
        "audit_rate_pct": audit_rate,
        "total_quarters": total_quarters,
        "audited_quarters": audited_quarters,
        "max_consecutive_unaudited": max_consec,
        "missing_rate_pct": missing_rate,
        "data_quality_flag": flag,
        "policy_keywords_found": policy_rows[:20],  # cap at 20 entries
        "notes": (
            f"~{audit_rate:.0f}% quý có kiểm toán đầy đủ — "
            "ưu tiên dùng Q4 (thường là quý kiểm toán năm). "
            "Dữ liệu chưa kiểm toán vẫn được dùng, chỉ gắn cờ cảnh báo."
        )
    }

    # Save JSON
    out_path = os.path.join(output_dir, 'preanalysis_report.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"  Audit Rate: {audit_rate:.1f}% ({audited_quarters}/{total_quarters} quý)")
    print(f"  Max Consecutive Unaudited: {max_consec} quý")
    print(f"  Missing Rate: {missing_rate:.2f}%")
    print(f"  Data Quality Flag: {flag}")
    print(f"  Policy Keywords Found: {len(policy_rows)} dòng")
    print(f"  Đã lưu: {out_path}")

    return report
