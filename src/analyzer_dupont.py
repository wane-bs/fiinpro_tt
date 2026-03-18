"""
analyzer_dupont.py
==================
Module 2.3 — DuPont 3-Step Analysis

Câu hỏi trả lời:
  ROE tăng/giảm là do biên lợi nhuận, hiệu quả tài sản hay đòn bẩy vốn?

Công thức:
  ROE = Net Profit Margin × Asset Turnover × Equity Multiplier

Output: output/dupont_analysis.csv
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


def compute_dupont(data_dict: dict, output_dir: str) -> pd.DataFrame:
    """
    Tính DuPont 3 nhân tố theo quý.
    ROE = Net Margin × Asset Turnover × Equity Multiplier
    """
    print("\n--- MODULE 2.3: DUPONT 3-STEP ANALYSIS ---")

    kqkd = data_dict.get('Kết quả kinh doanh')
    bctc = data_dict.get('Bảng cân đối kế toán')

    if kqkd is None or bctc is None:
        print("  [WARN] Thiếu KQKD hoặc BCĐKT — bỏ qua DuPont.")
        return pd.DataFrame()

    quarters = kqkd.columns[1:].tolist()

    # --- Trích dữ liệu ---
    net_revenue = _find_row(kqkd, 'Doanh thu thuần|Doanh thu thuan|Doanh thu bán hàng')
    net_income = _find_row(kqkd, 'Lãi/\(lỗ\) thuần sau thuế|Lợi nhuận sau thuế|Lợi nhuận thuần')
    tong_ts = _find_row(bctc, 'TỔNG CỘNG TÀI SẢN')
    von_chu = _find_row(bctc, 'VỐN CHỦ SỞ HỮU')

    if any(x is None for x in [net_revenue, net_income, tong_ts, von_chu]):
        print("  [WARN] Không tìm đủ dòng cần thiết cho DuPont.")
        return pd.DataFrame()

    rev_vals = net_revenue.values.astype(float)
    ni_vals = net_income.values.astype(float)
    ts_vals = tong_ts.values.astype(float)
    eq_vals = von_chu.values.astype(float)

    # Trung bình đầu/cuối kỳ
    avg_assets = (ts_vals + np.roll(ts_vals, 1)) / 2
    avg_equity = (eq_vals + np.roll(eq_vals, 1)) / 2
    avg_assets[0] = np.nan
    avg_equity[0] = np.nan

    # --- 3 nhân tố ---
    net_margin = np.where(rev_vals != 0, ni_vals / rev_vals, np.nan)
    asset_turn = np.where(avg_assets != 0, rev_vals / avg_assets, np.nan)
    equity_mult = np.where(avg_equity != 0, avg_assets / avg_equity, np.nan)

    # ROE DuPont (kiểm chứng)
    roe_dupont = net_margin * asset_turn * equity_mult * 100

    # ROE trực tiếp
    roe_direct = np.where(avg_equity != 0, ni_vals / avg_equity * 100, np.nan)

    # Delta ROE
    delta_roe = roe_dupont - np.roll(roe_dupont, 1)
    delta_roe[0] = np.nan

    # Build output DataFrame
    dupont_df = pd.DataFrame({
        'Quarter': quarters,
        'Net_Margin_Pct': np.round(net_margin * 100, 4),
        'Asset_Turnover': np.round(asset_turn, 4),
        'Equity_Multiplier': np.round(equity_mult, 4),
        'ROE_DuPont_Pct': np.round(roe_dupont, 4),
        'ROE_Direct_Pct': np.round(roe_direct, 4),
        'Delta_ROE': np.round(delta_roe, 4),
    })

    out_path = os.path.join(output_dir, 'dupont_analysis.csv')
    dupont_df.to_csv(out_path, index=False)

    # Print summary
    last = dupont_df.iloc[-1]
    print(f"  [Latest Quarter: {last['Quarter']}]")
    print(f"    Net Margin:       {last['Net_Margin_Pct']:.2f}%")
    print(f"    Asset Turnover:   {last['Asset_Turnover']:.4f}")
    print(f"    Equity Multiplier:{last['Equity_Multiplier']:.4f}")
    roe_dp = last['ROE_DuPont_Pct']
    roe_dr = last['ROE_Direct_Pct']
    d_roe = last['Delta_ROE']
    print(f"    ROE (DuPont):     {roe_dp:.2f}%" if not pd.isna(roe_dp) else "    ROE (DuPont):     N/A")
    print(f"    ROE (Direct):     {roe_dr:.2f}%" if not pd.isna(roe_dr) else "    ROE (Direct):     N/A")
    print(f"    Delta ROE:        {d_roe:+.2f}%" if not pd.isna(d_roe) else "    Delta ROE:        N/A")
    print(f"  Đã lưu: {out_path}")

    return dupont_df


def run_dupont_analysis(data_dict: dict, output_dir: str) -> pd.DataFrame:
    """Entry point cho Module 2.3 DuPont."""
    return compute_dupont(data_dict, output_dir)
