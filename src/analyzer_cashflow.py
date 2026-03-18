"""
analyzer_cashflow.py
====================
Module 2.4 — Cash Flow Quality Analysis

Câu hỏi trả lời:
  Lợi nhuận trên sổ sách có thực sự được hỗ trợ bởi tiền mặt?
  Công ty có dùng bút toán kế toán để thổi phồng lợi nhuận không?

Ngưỡng giải thích:
  - CFO/NI > 1.0: Chất lượng lợi nhuận cao
  - CFO/NI < 0.7 liên tiếp 3Q: Cảnh báo lợi nhuận ảo
  - Accrual Ratio > 5%: Tín hiệu bút toán kế toán

Output: output/cashflow_quality.csv
"""

import pandas as pd
import numpy as np
import os


def _find_row(df: pd.DataFrame, keyword: str, col_idx: int = 0):
    """Tìm dòng chứa keyword, trả về Series giá trị (bỏ cột tên)."""
    col = df.columns[col_idx]
    mask = df[col].str.contains(keyword, case=False, na=False, regex=True)
    if not mask.any():
        return None
    return df.loc[mask.idxmax(), df.columns[1:]].astype(float)


def compute_cashflow_quality(data_dict: dict, output_dir: str) -> pd.DataFrame:
    """
    Tính Cash Flow Quality metrics theo quý.
    CFO/NI, FCF, FCF Yield, Accrual Ratio + tự động gắn cờ cảnh báo.
    """
    print("\n--- MODULE 2.4: CASH FLOW QUALITY ---")

    lctt = data_dict.get('Lưu chuyển tiền tệ')
    kqkd = data_dict.get('Kết quả kinh doanh')
    bctc = data_dict.get('Bảng cân đối kế toán')

    if lctt is None:
        print("  [WARN] Thiếu sheet LCTT — bỏ qua Cash Flow Quality.")
        return pd.DataFrame()

    quarters = lctt.columns[1:].tolist()

    # --- Trích dữ liệu từ LCTT ---
    cfo = _find_row(lctt, 'Lưu chuyển tiền.*(?:kinh doanh|sản xuất)')
    cfi = _find_row(lctt, 'Lưu chuyển tiền.*(?:đầu tư)')
    cff = _find_row(lctt, 'Lưu chuyển tiền.*(?:tài chính)')
    capex = _find_row(lctt, 'mua sắm.*TSCĐ|mua sắm.*tài sản cố định|Chi mua TSCĐ')
    khau_hao = _find_row(lctt, 'Khấu hao TSCĐ')

    # --- Trích từ KQKD / BCĐKT ---
    net_income = _find_row(kqkd, 'Lãi/\(lỗ\) thuần sau thuế|Lợi nhuận sau thuế|Lợi nhuận thuần') if kqkd is not None else None
    net_revenue = _find_row(kqkd, 'Doanh thu thuần|Doanh thu thuan|Doanh thu bán hàng') if kqkd is not None else None
    tong_ts = _find_row(bctc, 'TỔNG CỘNG TÀI SẢN') if bctc is not None else None

    if cfo is None:
        print("  [WARN] Không tìm thấy dòng CFO trong LCTT.")
        return pd.DataFrame()

    cfo_vals = cfo.values.astype(float)

    # --- Build output ---
    result = pd.DataFrame({'Quarter': quarters})
    result['CFO'] = cfo_vals

    if cfi is not None:
        result['CFI'] = cfi.values.astype(float)
    else:
        result['CFI'] = np.nan

    if cff is not None:
        result['CFF'] = cff.values.astype(float)
    else:
        result['CFF'] = np.nan

    # FCF = CFO - |Capex|
    if capex is not None:
        capex_vals = np.abs(capex.values.astype(float))
        result['FCF'] = cfo_vals - capex_vals
    else:
        result['FCF'] = np.nan

    # CFO / NI Ratio
    if net_income is not None:
        ni_vals = net_income.values.astype(float)
        result['CFO_NI_Ratio'] = np.where(
            np.abs(ni_vals) > 0, cfo_vals / ni_vals, np.nan
        )
    else:
        result['CFO_NI_Ratio'] = np.nan

    # FCF Yield = FCF / Net Revenue * 100%
    if net_revenue is not None and capex is not None:
        rev_vals = net_revenue.values.astype(float)
        fcf_vals = result['FCF'].values
        result['FCF_Yield_Pct'] = np.where(
            rev_vals != 0, fcf_vals / rev_vals * 100, np.nan
        )
    else:
        result['FCF_Yield_Pct'] = np.nan

    # Accrual Ratio = (Net Income - CFO) / Avg Total Assets
    if net_income is not None and tong_ts is not None:
        ts_vals = tong_ts.values.astype(float)
        avg_assets = (ts_vals + np.roll(ts_vals, 1)) / 2
        avg_assets[0] = np.nan
        ni_vals = net_income.values.astype(float)
        result['Accrual_Ratio'] = np.where(
            avg_assets != 0, (ni_vals - cfo_vals) / avg_assets, np.nan
        )
    else:
        result['Accrual_Ratio'] = np.nan

    # --- Cảnh báo tự động ---
    flags = []
    cfo_ni = result['CFO_NI_Ratio'].values
    for i in range(len(quarters)):
        flag_list = []
        if not np.isnan(cfo_ni[i]) and cfo_ni[i] < 0.7:
            flag_list.append(f"CFO/NI={cfo_ni[i]:.2f}<0.7")
        accrual = result['Accrual_Ratio'].values[i] if 'Accrual_Ratio' in result.columns else np.nan
        if not np.isnan(accrual) and accrual > 0.05:
            flag_list.append(f"Accrual={accrual:.2%}>5%")
        flags.append('; '.join(flag_list) if flag_list else '')

    result['Flag'] = flags

    # Làm tròn
    numeric_cols = result.select_dtypes(include=np.number).columns
    result[numeric_cols] = result[numeric_cols].round(4)

    out_path = os.path.join(output_dir, 'cashflow_quality.csv')
    result.to_csv(out_path, index=False, encoding='utf-8-sig')

    # Print summary
    last = result.iloc[-1]
    print(f"  [Latest Quarter: {last['Quarter']}]")
    for col in ['CFO', 'FCF', 'CFO_NI_Ratio', 'FCF_Yield_Pct', 'Accrual_Ratio']:
        val = last.get(col, np.nan)
        if not pd.isna(val):
            print(f"    {col}: {val:.4f}")

    # Count warnings
    n_warn = sum(1 for f in flags if f)
    print(f"  Cảnh báo: {n_warn}/{len(quarters)} quý có flag bất thường")
    print(f"  Đã lưu: {out_path}")

    return result


def run_cashflow_analysis(data_dict: dict, output_dir: str) -> pd.DataFrame:
    """Entry point cho Module 2.4 Cash Flow Quality."""
    return compute_cashflow_quality(data_dict, output_dir)
