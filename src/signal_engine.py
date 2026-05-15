"""
signal_engine.py
================
Tru cot 4: Multi-Signal Composite Score (FPT Upgraded v2)

Composite Score = 3 trụ cột gắn vào output 3 chương:
  1. Sức khỏe tài chính (Chương 1): 30%
  2. Tăng trưởng & Chu kỳ (Chương 2): 35%
  3. Khoảng cách định giá (Chương 3): 35%

Score nằm trong khoảng [-100, +100].
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


def _load_csv(output_dir, filename):
    path = os.path.join(output_dir, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def _load_json(output_dir, filename):
    path = os.path.join(output_dir, filename)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


# ── Pillar 1: Financial Health (Chương 1) — 30% ──────────────────────

def _score_health(ratios_df, preanalysis):
    """
    Score from Current Ratio, D/E Ratio, and Audit Rate.
    Each sub-score: max ±100, then averaged.
    """
    scores = []

    # 1a. Current Ratio: >1.5 → +100, <0.8 → -100
    if not ratios_df.empty and 'Current_Ratio' in ratios_df.columns:
        cr = ratios_df['Current_Ratio'].iloc[-1]
        if not pd.isna(cr):
            s = max(-100, min(100, (cr - 1.0) * 200))
            scores.append(s)

    # 1b. D/E Ratio: <1.0 → +100, >3.0 → -100
    if not ratios_df.empty and 'DE_Ratio' in ratios_df.columns:
        de = ratios_df['DE_Ratio'].iloc[-1]
        if not pd.isna(de):
            s = max(-100, min(100, (2.0 - de) * 100))
            scores.append(s)

    # 1c. Audit Rate: >50% → positive
    if preanalysis:
        audit_rate = preanalysis.get('audit_rate_pct', 0)
        s = max(-100, min(100, (audit_rate - 50) * 2))
        scores.append(s)

    return float(np.mean(scores)) if scores else 0


# ── Pillar 2: Growth & Cycle (Chương 2) — 35% ────────────────────────

def _score_growth(ratios_df, dupont_df, cycle_df):
    """
    Score from Revenue Momentum, DuPont ΔROE, and Cycle Position.
    """
    scores = []

    # 2a. Revenue Momentum (3Q rolling average QoQ)
    if not ratios_df.empty and 'Rev_Momentum_3Q' in ratios_df.columns:
        mom = ratios_df['Rev_Momentum_3Q'].iloc[-1]
        if not pd.isna(mom):
            s = max(-100, min(100, (mom - 5) * 10))
            scores.append(s)

    # 2b. DuPont ΔROE: positive ΔRoE → bullish
    if not dupont_df.empty and 'Delta_ROE' in dupont_df.columns:
        delta_roe = dupont_df['Delta_ROE'].dropna()
        if len(delta_roe) > 0:
            recent_delta = delta_roe.iloc[-1]
            s = max(-100, min(100, recent_delta * 20))  # 5% → +100
            scores.append(s)

    # 2c. Cycle Position: Z-score of Trend
    if not cycle_df.empty and 'Trend' in cycle_df.columns:
        trends = cycle_df['Trend'].dropna()
        if len(trends) >= 4:
            latest_trend = trends.iloc[-1]
            mean_t = trends.mean()
            std_t = trends.std()
            if std_t > 0:
                z = (latest_trend - mean_t) / std_t
                s = max(-100, min(100, z * 40))
                scores.append(s)

    return float(np.mean(scores)) if scores else 0


# ── Pillar 3: Valuation Gap (Chương 3) — 35% ─────────────────────────

def _score_valuation(bands_df, sotp_scenarios, pe_forward_df, price_df):
    """
    Score from Mean Reversion Band Position, SoTP Upside, P/E Forward vs Hist.
    """
    scores = []
    price_now = None

    # Get current price
    if not price_df.empty:
        if 'TB_Gia_Ngay' in price_df.columns:
            price_now = price_df['TB_Gia_Ngay'].iloc[-1]
        elif 'Price' in price_df.columns:
            price_now = price_df['Price'].iloc[-1]

    # 3a. Mean Reversion Band Position
    if not bands_df.empty and 'Band_Position' in bands_df.columns:
        pos = bands_df['Band_Position'].iloc[-1]
        if not pd.isna(pos):
            s = (0.5 - pos) * 200  # 0=undervalued→+100, 1=overvalued→-100
            scores.append(max(-100, min(100, s)))

    # 3b. SoTP Upside (if available)
    if not sotp_scenarios.empty and price_now is not None and price_now > 0:
        base_row = sotp_scenarios[sotp_scenarios['Scenario'] == 'Base']
        if not base_row.empty:
            fair = base_row['Fair_Price_VND'].iloc[0]
            if not pd.isna(fair) and fair > 0:
                upside = (fair / price_now - 1) * 100
                s = max(-100, min(100, upside * 2))  # 50% upside → +100
                scores.append(s)

    # 3c. P/E Forward Band Position
    if not pe_forward_df.empty and 'PE_Band_Position' in pe_forward_df.columns:
        hist_rows = pe_forward_df[pe_forward_df['is_forward'] == False]
        if not hist_rows.empty:
            pe_pos = hist_rows['PE_Band_Position'].iloc[-1]
            if not pd.isna(pe_pos):
                s = (0.5 - pe_pos) * 200
                scores.append(max(-100, min(100, s)))

    return float(np.mean(scores)) if scores else 0


# ── Entry Point ───────────────────────────────────────────────────────

def generate_composite_signal(output_dir: str):
    """Tính toán Composite Score từ 3 trụ cột (Chương 1 + 2 + 3)."""
    print("\n--- COMPOSITE SCORE v2: 3 TRỤ CỘT ---")

    # Load all required data
    price_df = _load_csv(output_dir, 'aggregated_price_by_quarter.csv')
    ratios_df = _load_csv(output_dir, 'financial_ratios.csv')
    bands_df = _load_csv(output_dir, 'valuation_bands.csv')
    dupont_df = _load_csv(output_dir, 'dupont_analysis.csv')
    cycle_df = _load_csv(output_dir, 'cycle_decomposition.csv')
    sotp_scenarios = _load_csv(output_dir, 'sotp_scenarios.csv')
    pe_forward_df = _load_csv(output_dir, 'pe_forward_band.csv')
    preanalysis = _load_json(output_dir, 'preanalysis_report.json')

    if price_df.empty:
        print("  [ERROR] Thiếu aggregated_price_by_quarter.csv")
        return None

    last_q = (price_df.iloc[-1]['BCTC_Quarter_Label']
              if 'BCTC_Quarter_Label' in price_df.columns
              else price_df.iloc[-1].get('Quarter', 'N/A'))

    # ── Score each pillar ──
    s_health = _score_health(ratios_df, preanalysis)
    s_growth = _score_growth(ratios_df, dupont_df, cycle_df)
    s_valuation = _score_valuation(bands_df, sotp_scenarios, pe_forward_df, price_df)

    # Weights
    w = {
        'Health': 0.30,
        'Growth': 0.35,
        'Valuation': 0.35,
    }

    composite_score = (
        s_health * w['Health'] +
        s_growth * w['Growth'] +
        s_valuation * w['Valuation']
    )
    composite_score = round(composite_score, 1)

    # ── Verdict ──
    if composite_score >= 40:
        verdict = "MUA MẠNH"
        icon = "🟢"
    elif composite_score >= 15:
        verdict = "MUA (TÍCH LŨY)"
        icon = "🟢"
    elif composite_score >= -15:
        verdict = "TRUNG LẬP"
        icon = "🟡"
    elif composite_score >= -40:
        verdict = "BÁN"
        icon = "🔴"
    else:
        verdict = "BÁN MẠNH"
        icon = "🔴"

    details = {
        'Quarter': last_q,
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Composite_Score': composite_score,
        'Verdict': verdict,
        'Components': {
            'Health_Score': round(s_health, 1),
            'Growth_Score': round(s_growth, 1),
            'Valuation_Score': round(s_valuation, 1),
        },
        'Weights': w,
        'Version': 'v2_3pillars',
    }

    # Save JSON
    json_path = os.path.join(output_dir, 'recommendation.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(details, f, indent=4, ensure_ascii=False)

    # Save CSV history
    csv_path = os.path.join(output_dir, 'composite_signal.csv')
    rec_df = pd.DataFrame([{
        'Quarter': last_q,
        'Composite_Score': composite_score,
        'Health_Score': round(s_health, 1),
        'Growth_Score': round(s_growth, 1),
        'Valuation_Score': round(s_valuation, 1),
        'Verdict': verdict,
    }])
    if os.path.exists(csv_path):
        old_df = pd.read_csv(csv_path)
        old_df = old_df[old_df['Quarter'] != last_q]
        rec_df = pd.concat([old_df, rec_df], ignore_index=True)
    rec_df.to_csv(csv_path, index=False)

    print(f"  [TRỤ CỘT 1 — SỨC KHỎE]:    {s_health:+6.1f} × {w['Health']:.0%} = {s_health * w['Health']:+.1f}")
    print(f"  [TRỤ CỘT 2 — TĂNG TRƯỞNG]:  {s_growth:+6.1f} × {w['Growth']:.0%} = {s_growth * w['Growth']:+.1f}")
    print(f"  [TRỤ CỘT 3 — ĐỊNH GIÁ]:     {s_valuation:+6.1f} × {w['Valuation']:.0%} = {s_valuation * w['Valuation']:+.1f}")
    print(f"  ═══════════════════════════════════════")
    print(f"  [COMPOSITE SCORE]: {composite_score:+.1f} / 100")
    print(f"  [KHUYẾN NGHỊ]:     {icon} {verdict}")
    print(f"  Đã lưu: recommendation.json, composite_signal.csv")

    return details


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    generate_composite_signal(os.path.join(project_root, 'output'))
