"""
analyzer_ratios.py
==================
Tru cot 2: Financial Ratios Engine
Tinh cac chi so tai chinh tu BCTC co san trong data dict.

Chi so tinh:
  - Revenue CAGR 3Y (12 quy)
  - Gross Margin %
  - Operating Margin % (uoc tinh)
  - Revenue Momentum (rolling 3Q avg growth)
  - Short-term Debt Ratio
  - Revenue Acceleration (momentum cua momentum)
  - Current Ratio, Quick Ratio (Module 1.4)
  - D/E Ratio, Debt Ratio, Interest Coverage (Module 1.4)

Module 1.3: Vertical Analysis (BCĐKT + KQKD)

Output: output/financial_ratios.csv, output/vertical_analysis.csv
"""

import pandas as pd
import numpy as np
import os


def _find_row(df: pd.DataFrame, keyword: str, col_idx: int = 0):
    """Tim dong chua keyword, tra ve Series gia tri (bo cot ten)."""
    col = df.columns[col_idx]
    mask = df[col].str.contains(keyword, case=False, na=False, regex=True)
    if not mask.any():
        return None
    return df.loc[mask.idxmax(), df.columns[1:]].astype(float)


# ── Module 1.3: Vertical Analysis ─────────────────────────────────

def compute_vertical_analysis(data_dict: dict, output_dir: str) -> pd.DataFrame:
    """
    Module 1.3 — Vertical Analysis: tỷ trọng % theo quý.
    BCĐKT: % of TỔNG CỘNG TÀI SẢN
    KQKD: % of Doanh thu thuần
    Output: output/vertical_analysis.csv
    """
    print("\n--- MODULE 1.3: VERTICAL ANALYSIS ---")

    bctc = data_dict.get('Bảng cân đối kế toán')
    kqkd = data_dict.get('Kết quả kinh doanh')
    quarters = bctc.columns[1:].tolist() if bctc is not None else []

    rows = []

    # --- BCĐKT Vertical: Mảng Tài sản (mẫu số = Tổng tài sản) ---
    if bctc is not None:
        total_assets = _find_row(bctc, 'TỔNG CỘNG TÀI SẢN')
        total_capital = _find_row(bctc, 'TỔNG CỘNG NGUỒN VỐN')

        # Mảng Tài sản → Mẫu_số = 'Tổng tài sản'
        if total_assets is not None:
            ta_vals = total_assets.values.astype(float)
            asset_items = [
                ('Tiền và tương đương', 'BCĐKT'),
                ('Hàng tồn kho', 'BCĐKT'),
                ('Tài sản cố định', 'BCĐKT'),
                ('Phải thu ngắn hạn|Phải thu', 'BCĐKT'),
                ('TÀI SẢN NGẮN HẠN', 'BCĐKT'),
            ]
            for keyword, source in asset_items:
                row_data = _find_row(bctc, keyword)
                if row_data is not None:
                    pct = np.where(ta_vals != 0,
                                   row_data.values.astype(float) / ta_vals * 100,
                                   np.nan)
                    rows.append({'Chỉ_tiêu': keyword.split('|')[0],
                                 'Báo_cáo': source,
                                 'Mẫu_số': 'Tổng tài sản',
                                 **dict(zip(quarters, np.round(pct, 2)))})

        # Mảng Nguồn vốn → Mẫu_số = 'Tổng nguồn vốn'
        nv_vals = total_capital.values.astype(float) if total_capital is not None else (
            total_assets.values.astype(float) if total_assets is not None else None
        )
        if nv_vals is not None:
            capital_items = [
                ('NỢ NGẮN HẠN', 'BCĐKT'),
                ('NỢ PHẢI TRẢ', 'BCĐKT'),
                ('VỐN CHỦ SỞ HỮU', 'BCĐKT'),
            ]
            for keyword, source in capital_items:
                row_data = _find_row(bctc, keyword)
                if row_data is not None:
                    pct = np.where(nv_vals != 0,
                                   row_data.values.astype(float) / nv_vals * 100,
                                   np.nan)
                    rows.append({'Chỉ_tiêu': keyword.split('|')[0],
                                 'Báo_cáo': source,
                                 'Mẫu_số': 'Tổng nguồn vốn',
                                 **dict(zip(quarters, np.round(pct, 2)))})

    # --- KQKD Vertical (mẫu số = Doanh thu thuần) ---
    if kqkd is not None:
        net_rev = _find_row(kqkd, 'Doanh thu thuan|Doanh thu bán hàng')
        if net_rev is not None:
            rev_vals = net_rev.values.astype(float)
            kqkd_items = [
                ('Giá vốn', 'KQKD'),
                ('Chi phí bán hàng|Chi phi ban hang', 'KQKD'),
                ('Chi phí quản lý|Chi phi quan ly|Chi phí QLDN', 'KQKD'),
                ('Lợi nhuận thuần|Lợi nhuận sau thuế', 'KQKD'),
            ]
            for keyword, source in kqkd_items:
                row_data = _find_row(kqkd, keyword)
                if row_data is not None:
                    pct = np.where(rev_vals != 0,
                                   row_data.values.astype(float) / rev_vals * 100,
                                   np.nan)
                    rows.append({'Chỉ_tiêu': keyword.split('|')[0],
                                 'Báo_cáo': source,
                                 'Mẫu_số': 'Doanh thu thuần',
                                 **dict(zip(quarters, np.round(pct, 2)))})

    if not rows:
        print("  [WARN] Không tạo được Vertical Analysis.")
        return pd.DataFrame()

    vertical_df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, 'vertical_analysis.csv')
    vertical_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"  Đã lưu Vertical Analysis ({len(rows)} chỉ tiêu): {out_path}")
    return vertical_df


# ── Module 1.4 + Existing Ratios ──────────────────────────────────

def compute_financial_ratios(data_dict: dict, output_dir: str) -> pd.DataFrame:
    """
    Tinh Financial Ratios tu data_dict.
    Tra ve DataFrame theo quy voi cac chi so.
    """
    print("\n--- LUONG 5: FINANCIAL RATIOS ENGINE ---")

    kqkd = data_dict.get('Ket qua kinh doanh') or data_dict.get('Kết quả kinh doanh')
    bctc = data_dict.get('Bang can doi ke toan') or data_dict.get('Bảng cân đối kế toán')

    if kqkd is None:
        print("  [WARN] Khong tim thay sheet KQKD.")
        return pd.DataFrame()

    quarters = kqkd.columns[1:].tolist()
    ratios = pd.DataFrame({'Quarter': quarters})

    # --- Doanh thu thuan ---
    rev_net = _find_row(kqkd, 'Doanh thu thuan|Doanh thu bán hàng')
    if rev_net is not None:
        rev_vals = rev_net.values.astype(float)
        ratios['Revenue'] = rev_vals

        # QoQ Growth
        rev_s = pd.Series(rev_vals)
        ratios['Rev_QoQ_Pct'] = rev_s.pct_change() * 100

        # Revenue Momentum (3Q rolling avg of QoQ)
        ratios['Rev_Momentum_3Q'] = ratios['Rev_QoQ_Pct'].rolling(3).mean()

        # Revenue Acceleration (momentum cua momentum)
        ratios['Rev_Acceleration'] = ratios['Rev_Momentum_3Q'].diff()

        # CAGR 12Q (3 nam)
        cagr_list = [np.nan] * len(rev_vals)
        for i in range(12, len(rev_vals)):
            base = rev_vals[i - 12]
            curr = rev_vals[i]
            if base > 0 and curr > 0:
                cagr_list[i] = ((curr / base) ** (1 / 3) - 1) * 100
        ratios['Rev_CAGR_3Y_Pct'] = cagr_list

    # --- Gia von hang ban (GVHB) ---
    gvhb = _find_row(kqkd, 'Gia von|Giá vốn')
    if gvhb is not None and rev_net is not None:
        gross_profit = rev_net.values - gvhb.values
        ratios['Gross_Margin_Pct'] = np.where(
            rev_net.values != 0, gross_profit / rev_net.values * 100, np.nan
        )
    else:
        ratios['Gross_Margin_Pct'] = np.nan

    # --- Chi phi ban hang + quan ly -> uoc tinh EBIT ---
    cp_ban = _find_row(kqkd, 'Chi phi ban hang|Chi phí bán hàng')
    cp_qly = _find_row(kqkd, 'Chi phi quan ly|Chi phí quản lý|Chi phí QLDN')
    if cp_ban is not None and cp_qly is not None and rev_net is not None and gvhb is not None:
        ebit = rev_net.values - gvhb.values - cp_ban.values - cp_qly.values
        ratios['Operating_Margin_Pct'] = np.where(
            rev_net.values != 0, ebit / rev_net.values * 100, np.nan
        )
    else:
        ratios['Operating_Margin_Pct'] = np.nan

    # --- Short-term Debt Ratio (tu BCDK) ---
    if bctc is not None:
        vay_ngan = _find_row(bctc, 'Vay ngan han|Vay ngắn hạn')
        tong_nv = _find_row(bctc, 'TONG CONG NGUON VON|TỔNG CỘNG NGUỒN VỐN')
        if vay_ngan is not None and tong_nv is not None:
            ratios['ShortDebt_Ratio_Pct'] = np.where(
                tong_nv.values != 0, vay_ngan.values / tong_nv.values * 100, np.nan
            )
        else:
            ratios['ShortDebt_Ratio_Pct'] = np.nan
    else:
        ratios['ShortDebt_Ratio_Pct'] = np.nan

    # ── MODULE 1.4: Solvency & Liquidity Ratios ──────────────────
    if bctc is not None:
        tsnh = _find_row(bctc, 'TÀI SẢN NGẮN HẠN')
        nnh = _find_row(bctc, 'NỢ NGẮN HẠN')
        htk = _find_row(bctc, 'Hàng tồn kho')
        tong_no = _find_row(bctc, 'NỢ PHẢI TRẢ')
        von_chu = _find_row(bctc, 'VỐN CHỦ SỞ HỮU')
        tong_ts = _find_row(bctc, 'TỔNG CỘNG TÀI SẢN')

        # Current Ratio = TSNH / NNH
        if tsnh is not None and nnh is not None:
            nnh_vals = nnh.values.astype(float)
            ratios['Current_Ratio'] = np.where(
                nnh_vals != 0, tsnh.values.astype(float) / nnh_vals, np.nan
            )
        else:
            ratios['Current_Ratio'] = np.nan

        # Quick Ratio = (TSNH - HTK) / NNH
        if tsnh is not None and nnh is not None and htk is not None:
            nnh_vals = nnh.values.astype(float)
            ratios['Quick_Ratio'] = np.where(
                nnh_vals != 0,
                (tsnh.values.astype(float) - htk.values.astype(float)) / nnh_vals,
                np.nan
            )
        else:
            ratios['Quick_Ratio'] = np.nan

        # D/E Ratio = Tổng Nợ / Vốn CSH
        if tong_no is not None and von_chu is not None:
            vcsh_vals = von_chu.values.astype(float)
            ratios['DE_Ratio'] = np.where(
                vcsh_vals != 0, tong_no.values.astype(float) / vcsh_vals, np.nan
            )
        else:
            ratios['DE_Ratio'] = np.nan

        # Debt Ratio = Tổng Nợ / Tổng Tài Sản
        if tong_no is not None and tong_ts is not None:
            ts_vals = tong_ts.values.astype(float)
            ratios['Debt_Ratio'] = np.where(
                ts_vals != 0, tong_no.values.astype(float) / ts_vals, np.nan
            )
        else:
            ratios['Debt_Ratio'] = np.nan

        # Interest Coverage = EBIT / |Chi phí lãi vay|
        cp_lai_vay = _find_row(kqkd, 'Chi phí lãi vay|Chi phí tài chính|lãi vay')
        if cp_ban is not None and cp_qly is not None and rev_net is not None and gvhb is not None:
            ebit_vals = rev_net.values - gvhb.values - cp_ban.values - cp_qly.values
            if cp_lai_vay is not None:
                lai_vay_abs = np.abs(cp_lai_vay.values.astype(float))
                ratios['Interest_Coverage'] = np.where(
                    lai_vay_abs > 0, ebit_vals / lai_vay_abs, np.nan
                )
            else:
                ratios['Interest_Coverage'] = np.nan
        else:
            ratios['Interest_Coverage'] = np.nan
    else:
        ratios['Current_Ratio'] = np.nan
        ratios['Quick_Ratio'] = np.nan
        ratios['DE_Ratio'] = np.nan
        ratios['Debt_Ratio'] = np.nan
        ratios['Interest_Coverage'] = np.nan

    # ── MODULE 2.2: Activity & Profitability Ratios ─────────────────
    if bctc is not None and kqkd is not None:
        # Re-fetch rows needed (some may already exist from Module 1.4)
        tong_ts_22 = _find_row(bctc, 'TỔNG CỘNG TÀI SẢN')
        von_chu_22 = _find_row(bctc, 'VỐN CHỦ SỞ HỮU')
        htk_22 = _find_row(bctc, 'Hàng tồn kho')
        phai_thu = _find_row(bctc, 'Phải thu ngắn hạn|Phải thu')
        net_income = _find_row(kqkd, r'Lãi/\(lỗ\) thuần sau thuế|Lợi nhuận sau thuế|Lợi nhuận thuần')

        if tong_ts_22 is not None:
            ts_vals = tong_ts_22.values.astype(float)
            avg_assets = (ts_vals + np.roll(ts_vals, 1)) / 2
            avg_assets[0] = np.nan  # no prior period

            # Asset Turnover = Doanh thu / Avg Total Assets
            if rev_net is not None:
                ratios['Asset_Turnover'] = np.where(
                    avg_assets != 0, rev_net.values.astype(float) / avg_assets, np.nan
                )
            else:
                ratios['Asset_Turnover'] = np.nan

            # ROA = Net Income / Avg Total Assets * 100
            if net_income is not None:
                ratios['ROA_Pct'] = np.where(
                    avg_assets != 0, net_income.values.astype(float) / avg_assets * 100, np.nan
                )
            else:
                ratios['ROA_Pct'] = np.nan
        else:
            ratios['Asset_Turnover'] = np.nan
            ratios['ROA_Pct'] = np.nan

        if von_chu_22 is not None and net_income is not None:
            eq_vals = von_chu_22.values.astype(float)
            avg_equity = (eq_vals + np.roll(eq_vals, 1)) / 2
            avg_equity[0] = np.nan
            # ROE = Net Income / Avg Equity * 100
            ratios['ROE_Pct'] = np.where(
                avg_equity != 0, net_income.values.astype(float) / avg_equity * 100, np.nan
            )
        else:
            ratios['ROE_Pct'] = np.nan

        if htk_22 is not None and gvhb is not None:
            htk_vals = htk_22.values.astype(float)
            avg_inventory = (htk_vals + np.roll(htk_vals, 1)) / 2
            avg_inventory[0] = np.nan
            # Inventory Turnover = GVHB / Avg Inventory
            ratios['Inventory_Turnover'] = np.where(
                avg_inventory != 0, np.abs(gvhb.values.astype(float)) / avg_inventory, np.nan
            )
        else:
            ratios['Inventory_Turnover'] = np.nan

        if phai_thu is not None and rev_net is not None:
            pt_vals = phai_thu.values.astype(float)
            avg_receivable = (pt_vals + np.roll(pt_vals, 1)) / 2
            avg_receivable[0] = np.nan
            # Receivable Turnover = Doanh thu / Avg Receivable
            ratios['Receivable_Turnover'] = np.where(
                avg_receivable != 0, rev_net.values.astype(float) / avg_receivable, np.nan
            )
        else:
            ratios['Receivable_Turnover'] = np.nan
    else:
        ratios['Asset_Turnover'] = np.nan
        ratios['ROA_Pct'] = np.nan
        ratios['ROE_Pct'] = np.nan
        ratios['Inventory_Turnover'] = np.nan
        ratios['Receivable_Turnover'] = np.nan

    # --- Lam tron ---
    numeric_cols = ratios.select_dtypes(include=np.number).columns
    ratios[numeric_cols] = ratios[numeric_cols].round(4)

    out_path = os.path.join(output_dir, 'financial_ratios.csv')
    ratios.to_csv(out_path, index=False)
    print(f"  Da luu Financial Ratios: {out_path}")

    # In tom tat
    last = ratios.iloc[-1]
    print(f"  [Latest Quarter: {last.get('Quarter', '?')}]")
    for col in ['Gross_Margin_Pct', 'Operating_Margin_Pct', 'Rev_CAGR_3Y_Pct', 'ShortDebt_Ratio_Pct']:
        val = last.get(col, np.nan)
        if not pd.isna(val):
            print(f"    {col}: {val:.2f}%")

    return ratios


def run_vertical_analysis(data_dict: dict, output_dir: str) -> pd.DataFrame:
    """Entry point cho Module 1.3 Vertical Analysis."""
    return compute_vertical_analysis(data_dict, output_dir)


def run_ratio_analysis(data_dict: dict, output_dir: str) -> pd.DataFrame:
    """Entry point cho Luong 5 (includes Module 1.4 ratios)."""
    return compute_financial_ratios(data_dict, output_dir)
