"""
analyzer_sotp.py
================
Module 3.5: Sum-of-the-Parts (SoTP) Valuation for FPT Corporation.

Methodology:
  - Technology & Education: Valued using sector P/E multiples
  - Telecom: Valued using sector EV/EBITDA multiple
  - Total Equity = Sum(segment values) - Net Debt
  - 3 scenarios: Bear (-15%), Base, Bull (+15%)

Outputs:
  - output/sotp_valuation.csv
  - output/sotp_waterfall.csv (for Waterfall chart in dashboard)

Source: FPT Annual Report 2024, fpt.com.vn
"""

import pandas as pd
import numpy as np
import os

from segment_loader import SEGMENT_PROPORTIONS, IR_SOURCE


def _find_row_value(df: pd.DataFrame, keyword: str) -> float:
    """Find the last non-NaN value of a row matching keyword in col 0."""
    col0 = df.columns[0]
    mask = df[col0].str.contains(keyword, case=False, na=False, regex=True)
    if not mask.any():
        return np.nan
    row = df.loc[mask.idxmax()]
    vals = pd.to_numeric(row[df.columns[1:]], errors='coerce').dropna()
    return float(vals.iloc[-1]) if not vals.empty else np.nan


def run_sotp_valuation(data_dict: dict, segment_result: dict,
                       output_dir: str) -> dict:
    """
    Run Sum-of-the-Parts valuation.

    Parameters
    ----------
    data_dict : dict from data_loader
    segment_result : dict from segment_loader.load_segment_data()
    output_dir : str

    Returns
    -------
    dict with keys: sotp_df, waterfall_df, scenarios, source
    """
    print("\n--- MODULE 3.5: SUM-OF-THE-PARTS (SoTP) VALUATION ---")

    seg_df = segment_result.get('segments', pd.DataFrame())
    mode = segment_result.get('mode', 'estimated')
    source = segment_result.get('source', IR_SOURCE)

    print(f"  Chế độ dữ liệu: {mode.upper()}")
    print(f"  Nguồn: {source}")

    if seg_df.empty:
        print("  [WARN] Không có dữ liệu Segment — bỏ qua SoTP.")
        return {'sotp_df': pd.DataFrame(), 'waterfall_df': pd.DataFrame(),
                'scenarios': pd.DataFrame(), 'source': source}

    # Get latest quarter data
    latest_q = seg_df['Quarter'].iloc[-1]
    latest = seg_df[seg_df['Quarter'] == latest_q].copy()
    print(f"  Quý phân tích: {latest_q}")

    # Get consolidated data for Net Debt calculation
    bcdkt = data_dict.get('Bảng cân đối kế toán')
    chi_so = data_dict.get('chỉ số')

    # Net Debt = Total Borrowings - Cash & Equivalents
    net_debt = 0
    total_shares = 1  # billion shares, fallback

    if bcdkt is not None:
        vay_nh = _find_row_value(bcdkt, 'Vay và nợ thuê tài chính ngắn hạn|Vay ngắn hạn')
        vay_dh = _find_row_value(bcdkt, 'Vay và nợ thuê tài chính dài hạn|Vay dài hạn')
        tien = _find_row_value(bcdkt, 'Tiền và các khoản tương đương tiền')

        vay_nh = vay_nh if not np.isnan(vay_nh) else 0
        vay_dh = vay_dh if not np.isnan(vay_dh) else 0
        tien = tien if not np.isnan(tien) else 0

        net_debt = (vay_nh + vay_dh) - tien
        print(f"  Nợ vay ròng: {net_debt:,.0f} tỷ VND")

    # Estimate shares outstanding from Market Cap / Price
    if chi_so is not None:
        mkt_cap = _find_row_value(chi_so, 'Vốn hóa')
        eps_val = _find_row_value(chi_so, 'EPS cơ bản')
        if not np.isnan(mkt_cap) and not np.isnan(eps_val) and eps_val > 0:
            # Market cap in tỷ VND, EPS in VND
            # total_shares = market_cap * 1e9 / EPS (in VND) → result in shares
            # But we want in billion shares: total_shares = market_cap / (EPS / 1e9)
            total_shares = mkt_cap / (eps_val * 1e-9) * 1e-9  # billions of shares
            if total_shares <= 0:
                total_shares = 1.4  # FPT ~1.4 billion shares fallback

    # ── Segment Valuation ──
    valuations = []

    for _, row in latest.iterrows():
        seg = row['Segment']
        rev = row['Revenue']
        ebitda = row['EBITDA']
        ni = row.get('Net_Income', np.nan)
        params = SEGMENT_PROPORTIONS.get(seg, {})

        if 'peer_pe' in params and not np.isnan(ni if not pd.isna(ni) else np.nan):
            # P/E method: Value = NI * 4 (annualise quarterly) * Peer P/E
            annual_ni = ni * 4
            seg_value = annual_ni * params['peer_pe']
            method = f"P/E = {params['peer_pe']}"
        elif 'peer_ev_ebitda' in params:
            # EV/EBITDA method: EV = EBITDA * 4 * Peer EV/EBITDA
            annual_ebitda = ebitda * 4
            seg_value = annual_ebitda * params['peer_ev_ebitda']
            method = f"EV/EBITDA = {params['peer_ev_ebitda']}"
        else:
            # Fallback: Revenue multiple
            seg_value = rev * 4 * 2.0  # 2x revenue
            method = "P/S = 2.0 (fallback)"

        valuations.append({
            'Segment': seg,
            'Revenue_Q': round(rev, 2),
            'EBITDA_Q': round(ebitda, 2),
            'NI_Q': round(ni, 2) if not pd.isna(ni) else np.nan,
            'Method': method,
            'Segment_Value': round(seg_value, 2),
        })

    val_df = pd.DataFrame(valuations)
    total_ev = val_df['Segment_Value'].sum()
    equity_value = total_ev - net_debt

    # Per-share value (VND)
    # equity_value is in tỷ VND, total_shares in billion shares
    # fair_price = equity_value * 1e9 / (total_shares * 1e9) = equity_value / total_shares (VND)
    # But equity_value is in tỷ, so: fair_price = equity_value / total_shares * 1000 (VND)
    if total_shares > 0:
        fair_price_per_share = equity_value / total_shares * 1000
    else:
        fair_price_per_share = 0

    print(f"  Tổng EV (SoTP): {total_ev:,.0f} tỷ VND")
    print(f"  Equity Value: {equity_value:,.0f} tỷ VND")
    print(f"  Giá hợp lý/cp: {fair_price_per_share:,.0f} VND")

    # ── Scenarios ──
    scenarios = pd.DataFrame([
        {'Scenario': 'Bear', 'Multiple_Adj': 0.85,
         'Total_EV': round(total_ev * 0.85, 2),
         'Net_Debt': round(net_debt, 2),
         'Equity_Value': round(total_ev * 0.85 - net_debt, 2),
         'Fair_Price_VND': round(fair_price_per_share * 0.85, 0),
         'is_forward': True},
        {'Scenario': 'Base', 'Multiple_Adj': 1.00,
         'Total_EV': round(total_ev, 2),
         'Net_Debt': round(net_debt, 2),
         'Equity_Value': round(equity_value, 2),
         'Fair_Price_VND': round(fair_price_per_share, 0),
         'is_forward': True},
        {'Scenario': 'Bull', 'Multiple_Adj': 1.15,
         'Total_EV': round(total_ev * 1.15, 2),
         'Net_Debt': round(net_debt, 2),
         'Equity_Value': round(total_ev * 1.15 - net_debt, 2),
         'Fair_Price_VND': round(fair_price_per_share * 1.15, 0),
         'is_forward': True},
    ])

    # ── Waterfall data ──
    waterfall_rows = []
    for _, v in val_df.iterrows():
        waterfall_rows.append({
            'Component': v['Segment'],
            'Value': round(v['Segment_Value'], 2),
            'Type': 'segment',
        })
    waterfall_rows.append({
        'Component': 'Trừ Nợ vay ròng',
        'Value': round(-net_debt, 2),
        'Type': 'deduction',
    })
    waterfall_rows.append({
        'Component': 'Equity Value',
        'Value': round(equity_value, 2),
        'Type': 'total',
    })
    waterfall_df = pd.DataFrame(waterfall_rows)

    # ── Save outputs ──
    val_df.to_csv(os.path.join(output_dir, 'sotp_valuation.csv'), index=False)
    waterfall_df.to_csv(os.path.join(output_dir, 'sotp_waterfall.csv'), index=False)
    scenarios.to_csv(os.path.join(output_dir, 'sotp_scenarios.csv'), index=False)

    print(f"\n  === SoTP SCENARIOS ===")
    for _, row in scenarios.iterrows():
        icon = "📈" if row['Scenario'] == 'Bull' else ("📉" if row['Scenario'] == 'Bear' else "📊")
        print(f"  {icon} {row['Scenario']:4s}: {row['Fair_Price_VND']:>10,.0f} VND/cp "
              f"(EV: {row['Total_EV']:,.0f} tỷ)")
    print(f"  ⚠️  Dữ liệu ước tính từ IR — Nguồn: {source}")
    print(f"  Đã lưu: sotp_valuation.csv, sotp_waterfall.csv, sotp_scenarios.csv")

    return {
        'sotp_df': val_df,
        'waterfall_df': waterfall_df,
        'scenarios': scenarios,
        'source': source,
        'fair_price_base': fair_price_per_share,
        'equity_value': equity_value,
        'net_debt': net_debt,
        'mode': mode,
    }
