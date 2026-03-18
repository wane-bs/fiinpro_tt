"""
signal_engine.py
================
Tru cot 4: Multi-Signal Composite Score

Ket hop cac tin hieu de dua ra khuyen nghi MUA / BAN / TRUNG LAP cuoi cung.
Score nam trong khoang [-100, +100].

Cac thanh phan:
  1. Mua vu (Thang 4-5) [25%]
  2. Momentum Gia thi truong [20%]
  3. Revenue Momentum [20%]
  4. Vi tri Gia vs Valuation Band [25%]
  5. Volume Surge [10%]
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


def _score_seasonality(last_quarter_label):
    """
    Kiem tra neu dang o Q1 hoac Q2 -> huong vao vung mua vu T4-T5.
    Neu Q1 hoac Q2 -> Score 100.
    Neu Q3 hoac Q4 -> Score -50 (vi qua mua vu, rui ro dieu chinh).
    """
    try:
        q = int(last_quarter_label.split('/')[0][1])
        if q in [1, 2]:
            return 100
        else:
            return -50
    except:
        return 0


def _score_price_momentum(price_df):
    """
    Momentum = Log Return gan nhat (quy).
    Neu return >= 10% -> 100
    Neu return <= -10% -> -100
    Giao dong tuyen tinh o giua.
    """
    if price_df.empty or 'TB_Gia_Ngay' not in price_df.columns:
        return 0
    if len(price_df) >= 2:
        ret = price_df['TB_Gia_Ngay'].iloc[-1] / price_df['TB_Gia_Ngay'].iloc[-2] - 1
        score = max(-100, min(100, ret * 10))  # 10% -> 100
        return score
    return 0


def _score_revenue_momentum(ratios_df):
    """
    Dua tren Rev_Momentum_3Q (rolling avg QoQ).
    """
    if ratios_df.empty or 'Rev_Momentum_3Q' not in ratios_df.columns:
        return 0
    mom = ratios_df['Rev_Momentum_3Q'].iloc[-1]
    if pd.isna(mom):
        return 0
    # >10% QoQ avg -> 100, < 0% -> -100
    score = max(-100, min(100, (mom - 5) * 10))
    return float(score)


def _score_valuation_band(bands_df):
    """
    Vi tri gia so voi dải Mean Reversion.
    Band_Position: 0 = duoi Lower Band (UnderValued -> Score 100)
    1 = tren Upper Band (OverValued -> Score -100)
    0.5 = Fair Value (Score 0)
    """
    if bands_df.empty or 'Band_Position' not in bands_df.columns:
        return 0
    pos = bands_df['Band_Position'].iloc[-1]
    if pd.isna(pos):
        return 0
    # Map [0, 1] -> [100, -100]
    score = (0.5 - pos) * 200
    return max(-100, min(100, score))


def _score_volume_surge(price_df):
    """
    So sanh quy hien tai voi trung binh 4 quy truoc.
    """
    if price_df.empty or 'TB_KhoiLuong_KhopLenh_Ngay' not in price_df.columns:
        return 0
    if len(price_df) < 5:
        return 0
    vol_now = price_df['TB_KhoiLuong_KhopLenh_Ngay'].iloc[-1]
    vol_hist = price_df['TB_KhoiLuong_KhopLenh_Ngay'].iloc[-5:-1].mean()
    if vol_hist == 0:
        return 0
    surge = vol_now / vol_hist - 1
    # surge 50% -> score 100
    score = max(-100, min(100, surge * 200))
    return score


def generate_composite_signal(output_dir: str):
    """Tinh toan Composite Score tu nhieu ban tin hieu."""
    print("\n--- LUONG 7: MULTI-SIGNAL COMPOSITE SCORE ---")

    price_df = _load_csv(output_dir, 'aggregated_price_by_quarter.csv')
    ratios_df = _load_csv(output_dir, 'financial_ratios.csv')
    bands_df = _load_csv(output_dir, 'valuation_bands.csv')

    if price_df.empty:
        print("  [ERROR] Thieu aggregated_price_by_quarter.csv")
        return None

    last_q = price_df.iloc[-1]['BCTC_Quarter_Label'] if 'BCTC_Quarter_Label' in price_df.columns else price_df.iloc[-1]['Quarter']

    # --- Tinh toan tung phan ---
    s_season = _score_seasonality(last_q)
    s_price_mom = _score_price_momentum(price_df)
    s_rev_mom = _score_revenue_momentum(ratios_df)
    s_value = _score_valuation_band(bands_df)
    s_vol = _score_volume_surge(price_df)

    # Weights
    w = {
        'Seasonality': 0.25,
        'Price_Momentum': 0.20,
        'Revenue_Momentum': 0.20,
        'Valuation': 0.25,
        'Volume_Surge': 0.10
    }

    composite_score = (
        s_season * w['Seasonality'] +
        s_price_mom * w['Price_Momentum'] +
        s_rev_mom * w['Revenue_Momentum'] +
        s_value * w['Valuation'] +
        s_vol * w['Volume_Surge']
    )
    composite_score = round(composite_score, 1)

    # --- Ra quyet dinh ---
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
            'Seasonality_Score': round(s_season, 1),
            'Price_Momentum_Score': round(s_price_mom, 1),
            'Revenue_Momentum_Score': round(s_rev_mom, 1),
            'Valuation_Band_Score': round(s_value, 1),
            'Volume_Surge_Score': round(s_vol, 1)
        },
        'Weights': w
    }

    # Luu JSON
    json_path = os.path.join(output_dir, 'recommendation.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(details, f, indent=4, ensure_ascii=False)

    # Luu CSV history append (neu can trong ung dung sau nay)
    csv_path = os.path.join(output_dir, 'composite_signal.csv')
    rec_df = pd.DataFrame([{
        'Quarter': last_q,
        'Composite_Score': composite_score,
        'Seasonality_Score': round(s_season, 1),
        'Valuation_Score': round(s_value, 1),
        'Verdict': verdict
    }])
    if os.path.exists(csv_path):
        old_df = pd.read_csv(csv_path)
        # remove duplicate for same quarter
        old_df = old_df[old_df['Quarter'] != last_q]
        rec_df = pd.concat([old_df, rec_df], ignore_index=True)
    rec_df.to_csv(csv_path, index=False)

    print(f"  [COMPOSITE SCORE]: {composite_score:+.1f} / 100")
    print(f"  [KHUYEN NGHI HOAT DONG]: {icon} {verdict}")
    print(f"  Da luu: recommendation.json, composite_signal.csv")

    return details


if __name__ == "__main__":
    # Test doc lap
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    generate_composite_signal(os.path.join(project_root, 'output'))
