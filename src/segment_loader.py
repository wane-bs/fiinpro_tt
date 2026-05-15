"""
segment_loader.py
=================
Data Layer: Load/estimate FPT segment data (Technology, Telecom, Education).

Two modes:
  1. Real: Read 'Segment_Data' sheet from data.xlsx if available.
  2. Estimated: Apply IR 2024 revenue proportions onto consolidated KQKD.

Source: FPT Annual Report 2024, FPT 12M/2024 Earnings Release — fpt.com.vn
"""

import pandas as pd
import numpy as np

# ── FPT IR 2024 Segment Proportions ──────────────────────────────────
# Source: FPT 12M/2024 Earnings Release (fpt.com.vn/investor-relations)
# Technology: 39,110 / 62,849 = 62.2%
# Telecom:    16,906 / 62,849 = 26.9%
# Education:   7,088 / 62,849 = 11.3%
# Rounding: sum ~100.4% due to inter-segment elimination — re-normalised.

SEGMENT_PROPORTIONS = {
    'Technology': {
        'revenue_pct': 0.620,
        'ebitda_margin_est': 0.17,      # ~17% EBITDA margin (IT services)
        'capex_to_rev_est': 0.04,       # ~4% Capex/Revenue
        'peer_pe': 22.0,                # Sector P/E for IT services
        'yoy_growth_2024': 0.244,       # +24.4% YoY
    },
    'Telecom': {
        'revenue_pct': 0.270,
        'ebitda_margin_est': 0.35,      # ~35% EBITDA margin (infra/telecom)
        'capex_to_rev_est': 0.18,       # ~18% Capex/Revenue (heavy infra)
        'peer_ev_ebitda': 6.5,          # Sector EV/EBITDA for telecom
        'yoy_growth_2024': 0.113,       # +11.3% YoY
    },
    'Education': {
        'revenue_pct': 0.110,
        'ebitda_margin_est': 0.22,      # ~22% EBITDA margin (education)
        'capex_to_rev_est': 0.08,       # ~8% Capex/Revenue
        'peer_pe': 18.0,               # Sector P/E for education
        'yoy_growth_2024': 0.151,       # +15.1% YoY
    },
}

IR_SOURCE = "FPT Annual Report 2024, 12M/2024 Earnings Release — fpt.com.vn"


def load_segment_data(data_dict: dict, file_path: str = None) -> dict:
    """
    Attempt to load segment data. Returns a dict with:
      - 'segments': pd.DataFrame (Quarter, Segment, Revenue, EBITDA, ...)
      - 'mode': 'real' | 'estimated'
      - 'source': citation string
      - 'proportions': SEGMENT_PROPORTIONS used
    """

    # Mode 1: Try reading real Segment_Data sheet
    if file_path:
        try:
            xls = pd.ExcelFile(file_path)
            if 'Segment_Data' in xls.sheet_names:
                seg_df = pd.read_excel(xls, sheet_name='Segment_Data')
                return {
                    'segments': seg_df,
                    'mode': 'real',
                    'source': 'Sheet Segment_Data trong data.xlsx',
                    'proportions': None,
                }
        except Exception:
            pass

    # Mode 2: Estimate from consolidated KQKD
    return _estimate_segments(data_dict)


def _estimate_segments(data_dict: dict) -> dict:
    """
    Estimate segment-level Revenue & EBITDA from consolidated KQKD
    using FPT IR 2024 proportions.
    """
    kqkd = data_dict.get('Kết quả kinh doanh')
    if kqkd is None or kqkd.empty:
        return {
            'segments': pd.DataFrame(),
            'mode': 'estimated',
            'source': IR_SOURCE,
            'proportions': SEGMENT_PROPORTIONS,
        }

    # Find revenue row
    col0 = kqkd.columns[0]
    rev_mask = kqkd[col0].str.contains('Doanh thu thuần|Doanh thu bán hàng',
                                        case=False, na=False, regex=True)
    if not rev_mask.any():
        return {
            'segments': pd.DataFrame(),
            'mode': 'estimated',
            'source': IR_SOURCE,
            'proportions': SEGMENT_PROPORTIONS,
        }

    rev_row = kqkd.loc[rev_mask.idxmax()]
    quarters = kqkd.columns[1:]
    revenues = pd.to_numeric(rev_row[quarters], errors='coerce')

    # Find LNST (net income) for margin estimation
    ni_mask = kqkd[col0].str.contains('LN sau thuế.*cổ đông công ty mẹ|Lợi nhuận sau thuế',
                                       case=False, na=False, regex=True)
    if ni_mask.any():
        ni_row = kqkd.loc[ni_mask.idxmax()]
        net_incomes = pd.to_numeric(ni_row[quarters], errors='coerce')
    else:
        net_incomes = revenues * 0.10  # fallback 10% margin

    rows = []
    for q_label in quarters:
        rev_total = revenues.get(q_label, np.nan)
        ni_total = net_incomes.get(q_label, np.nan)

        if pd.isna(rev_total):
            continue

        for seg_name, params in SEGMENT_PROPORTIONS.items():
            seg_rev = rev_total * params['revenue_pct']
            seg_ebitda = seg_rev * params['ebitda_margin_est']
            seg_capex = seg_rev * params['capex_to_rev_est']
            seg_ni = ni_total * params['revenue_pct'] if not pd.isna(ni_total) else np.nan

            rows.append({
                'Quarter': q_label,
                'Segment': seg_name,
                'Revenue': round(seg_rev, 2),
                'EBITDA': round(seg_ebitda, 2),
                'Capex': round(seg_capex, 2),
                'Net_Income': round(seg_ni, 2) if not pd.isna(seg_ni) else np.nan,
                'is_estimated': True,
            })

    seg_df = pd.DataFrame(rows)

    return {
        'segments': seg_df,
        'mode': 'estimated',
        'source': IR_SOURCE,
        'proportions': SEGMENT_PROPORTIONS,
    }


def get_ir_summary_table() -> pd.DataFrame:
    """Return a summary table of FPT IR 2024 proportions for dashboard display."""
    rows = []
    for seg_name, params in SEGMENT_PROPORTIONS.items():
        rows.append({
            'Mảng kinh doanh': seg_name,
            'Tỷ trọng DT (%)': f"{params['revenue_pct'] * 100:.1f}%",
            'EBITDA Margin ước tính': f"{params['ebitda_margin_est'] * 100:.0f}%",
            'Tăng trưởng YoY 2024': f"+{params['yoy_growth_2024'] * 100:.1f}%",
            'Phương pháp định giá': (
                f"P/E = {params['peer_pe']}" if 'peer_pe' in params
                else f"EV/EBITDA = {params['peer_ev_ebitda']}"
            ),
        })
    return pd.DataFrame(rows)
