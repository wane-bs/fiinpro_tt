"""
analyzer_dcf.py
===============
Module 3.2 — DCF Valuation + WACC Sensitivity Analysis

Câu hỏi trả lời:
  Giá trị nội tại của doanh nghiệp là bao nhiêu?
  Nếu WACC hoặc tốc độ tăng trưởng thay đổi ±2%, giá trị thay đổi thế nào?

Output:
  output/dcf_valuation.csv    — WACC, g, Equity Value
  output/dcf_sensitivity.csv  — Ma trận độ nhạy WACC × g_terminal
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


def _compute_working_capital_change(bctc):
    """
    Tính Δ Working Capital = Δ(TSNH - Tiền - NNH).
    WC = TSNH - Tiền - NNH → ΔWC = WC_t - WC_{t-1}
    """
    tsnh = _find_row(bctc, 'TÀI SẢN NGẮN HẠN')
    tien = _find_row(bctc, 'Tiền và tương đương')
    nnh = _find_row(bctc, 'NỢ NGẮN HẠN')

    if any(x is None for x in [tsnh, tien, nnh]):
        return np.zeros(len(bctc.columns) - 1)

    wc = tsnh.values.astype(float) - tien.values.astype(float) - nnh.values.astype(float)
    delta_wc = np.diff(wc, prepend=wc[0])
    return delta_wc


def run_dcf_analysis(data_dict: dict, output_dir: str) -> dict:
    """
    Module 3.2 — DCF Valuation + Sensitivity.
    """
    print("\n--- MODULE 3.2: DCF VALUATION + SENSITIVITY ---")

    lctt = data_dict.get('Lưu chuyển tiền tệ')
    kqkd = data_dict.get('Kết quả kinh doanh')
    bctc = data_dict.get('Bảng cân đối kế toán')

    if any(x is None for x in [lctt, kqkd, bctc]):
        print("  [WARN] Thiếu LCTT/KQKD/BCĐKT — bỏ qua DCF.")
        return {}

    quarters = kqkd.columns[1:].tolist()

    # --- Step 1: FCFF lịch sử ---
    net_revenue = _find_row(kqkd, 'Doanh thu thuần|Doanh thu thuan|Doanh thu bán hàng')
    gvhb = _find_row(kqkd, 'Giá vốn|Gia von')
    cp_ban = _find_row(kqkd, 'Chi phí bán hàng|Chi phi ban hang')
    cp_qly = _find_row(kqkd, 'Chi phí quản lý|Chi phi quan ly|Chi phí QLDN')
    cp_lai_vay = _find_row(kqkd, 'Chi phí lãi vay|Chi phí tài chính|lãi vay')

    khau_hao = _find_row(lctt, 'Khấu hao TSCĐ')
    capex = _find_row(lctt, 'mua sắm.*TSCĐ|mua sắm.*tài sản cố định|Chi mua TSCĐ')
    tong_no = _find_row(bctc, 'NỢ PHẢI TRẢ')
    von_chu = _find_row(bctc, 'VỐN CHỦ SỞ HỮU')

    if any(x is None for x in [net_revenue, gvhb, tong_no, von_chu]):
        print("  [WARN] Không tìm đủ dòng cần thiết cho DCF.")
        return {}

    # EBIT estimate
    rev_vals = net_revenue.values.astype(float)
    gvhb_vals = np.abs(gvhb.values.astype(float))
    cp_ban_vals = np.abs(cp_ban.values.astype(float)) if cp_ban is not None else np.zeros_like(rev_vals)
    cp_qly_vals = np.abs(cp_qly.values.astype(float)) if cp_qly is not None else np.zeros_like(rev_vals)
    ebit_vals = rev_vals - gvhb_vals - cp_ban_vals - cp_qly_vals

    tax_rate = 0.20  # thuế TNDN Việt Nam ~20%
    nopat = ebit_vals * (1 - tax_rate)

    khau_hao_vals = khau_hao.values.astype(float) if khau_hao is not None else np.zeros_like(rev_vals)
    capex_vals = np.abs(capex.values.astype(float)) if capex is not None else np.zeros_like(rev_vals)
    delta_wc = _compute_working_capital_change(bctc)

    # Align lengths (LCTT may have different length than KQKD)
    n = min(len(nopat), len(khau_hao_vals), len(capex_vals), len(delta_wc))
    fcff_hist = nopat[:n] + khau_hao_vals[:n] - delta_wc[:n] - capex_vals[:n]

    # --- Step 2: WACC ---
    D = float(tong_no.values[-1])
    E = float(von_chu.values[-1])
    V = D + E

    if V <= 0:
        print("  [WARN] V = D + E <= 0 — bỏ qua DCF.")
        return {}

    # Cost of debt
    if cp_lai_vay is not None and D > 0:
        r_d = float(np.abs(cp_lai_vay.values[-1])) / D * 4  # annualize quarterly
    else:
        r_d = 0.08

    # Cost of equity (CAPM)
    r_f = 0.045   # TPCP 10 năm VN
    erp = 0.07    # Equity Risk Premium
    beta = 1.0
    r_e = r_f + beta * erp

    wacc = (E / V) * r_e + (D / V) * r_d * (1 - tax_rate)
    wacc = max(wacc, 0.05)  # floor at 5%

    # --- Step 3: DCF Projection ---
    # Use median of last 8 quarters as base FCFF
    valid_fcff = fcff_hist[~np.isnan(fcff_hist)]
    if len(valid_fcff) < 4:
        print("  [WARN] Không đủ dữ liệu FCFF lịch sử.")
        return {}

    fcff_base = float(np.nanmedian(valid_fcff[-8:]))

    # Annualize (quarterly → annual)
    fcff_base_annual = fcff_base * 4

    # Growth rate from 3-year CAGR
    if len(valid_fcff) >= 12:
        f_start = float(np.nanmean(valid_fcff[-12:-8]))  # avg year -3
        f_end = float(np.nanmean(valid_fcff[-4:]))        # avg last year
        if f_start > 0 and f_end > 0:
            g_growth = (f_end / f_start) ** (1 / 3) - 1
        else:
            g_growth = 0.05
    else:
        g_growth = 0.05

    g_growth = max(min(g_growth, 0.25), -0.10)  # cap growth

    g_tv = 0.03  # terminal growth rate

    # PV of projected FCFF (5 years)
    pv_fcff = sum(
        fcff_base_annual * (1 + g_growth) ** t / (1 + wacc) ** t
        for t in range(1, 6)
    )

    # Terminal Value
    fcff_year5 = fcff_base_annual * (1 + g_growth) ** 5
    if wacc > g_tv:
        tv = fcff_year5 * (1 + g_tv) / (wacc - g_tv)
        pv_tv = tv / (1 + wacc) ** 5
    else:
        pv_tv = 0

    enterprise_value = pv_fcff + pv_tv
    equity_value = enterprise_value - D

    # Save DCF summary
    dcf_summary = pd.DataFrame([{
        'WACC_Pct': round(wacc * 100, 2),
        'Growth_Pct': round(g_growth * 100, 2),
        'Terminal_Growth_Pct': round(g_tv * 100, 2),
        'FCFF_Base_Annual': round(fcff_base_annual, 2),
        'PV_FCFF_5Y': round(pv_fcff, 2),
        'PV_Terminal': round(pv_tv, 2),
        'Enterprise_Value': round(enterprise_value, 2),
        'Net_Debt': round(D, 2),
        'Equity_Value': round(equity_value, 2),
        'r_e_Pct': round(r_e * 100, 2),
        'r_d_Pct': round(r_d * 100, 2),
        'D_V_Pct': round(D / V * 100, 2),
        'E_V_Pct': round(E / V * 100, 2),
    }])

    dcf_path = os.path.join(output_dir, 'dcf_valuation.csv')
    dcf_summary.to_csv(dcf_path, index=False)

    # --- Step 4: Sensitivity Matrix ---
    wacc_range = np.arange(wacc - 0.02, wacc + 0.025, 0.005)
    g_range = np.arange(g_tv - 0.02, g_tv + 0.025, 0.005)

    rows = []
    for w in wacc_range:
        row = {'WACC': f"{w:.1%}"}
        for g in g_range:
            if w > g:
                pv_sum = sum(
                    fcff_base_annual * (1 + g_growth) ** t / (1 + w) ** t
                    for t in range(1, 6)
                )
                tv_s = fcff_year5 * (1 + g) / (w - g)
                pv_total = pv_sum + tv_s / (1 + w) ** 5 - D
                row[f"g={g:.1%}"] = round(pv_total, 0)
            else:
                row[f"g={g:.1%}"] = None
        rows.append(row)

    sensitivity_df = pd.DataFrame(rows)
    sens_path = os.path.join(output_dir, 'dcf_sensitivity.csv')
    sensitivity_df.to_csv(sens_path, index=False, encoding='utf-8-sig')

    # Print summary
    print(f"  WACC:             {wacc:.2%}")
    print(f"  FCFF Growth:      {g_growth:.2%}")
    print(f"  FCFF Base (年):   {fcff_base_annual:,.0f} tỷ VND")
    print(f"  Enterprise Value: {enterprise_value:,.0f} tỷ VND")
    print(f"  Equity Value:     {equity_value:,.0f} tỷ VND")
    print(f"  Sensitivity Grid: {len(wacc_range)}×{len(g_range)}")
    print(f"  Đã lưu: {dcf_path}, {sens_path}")

    return {
        'wacc': wacc,
        'g_growth': g_growth,
        'equity_value': equity_value,
        'enterprise_value': enterprise_value,
    }
