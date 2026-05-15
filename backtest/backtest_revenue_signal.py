"""
backtest_revenue_signal.py
==========================
Kiểm định: Tín hiệu doanh thu tăng trưởng mạnh (>+10% QoQ)
có dẫn dắt giá tăng trong 30/60/90 ngày tiếp theo không?

Phương pháp:
  1. Từ revenue_growth.csv: xác định các quý có DT_QoQ > threshold
  2. Từ aggregated_price_by_quarter.csv: lấy giá khớp nhãn quý
  3. Tính Forward Return (giá quý tiếp theo vs quý hiện tại)
  4. So sánh với Base Rate (tất cả các quý)
  5. Tính Hit Rate, Avg Forward Return, T-test significance
"""

import pandas as pd
import numpy as np
import os


def _load_revenue(output_dir: str) -> pd.DataFrame:
    path = os.path.join(output_dir, "revenue_growth.csv")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # Đảm bảo cột đúng tên
    df = df.rename(columns={
        'Quý': 'Quarter',
        'Doanh_Thu': 'Revenue',
        'Tang_Truong_QoQ_Pct': 'Growth_QoQ_Pct'
    })
    return df


def _load_price_by_quarter(output_dir: str) -> pd.DataFrame:
    path = os.path.join(output_dir, "aggregated_price_by_quarter.csv")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={'BCTC_Quarter_Label': 'Quarter'})
    return df


def _quarter_year(label: str):
    """'Q2/2023' -> (2023, 2)"""
    try:
        parts = label.strip().split('/')
        q = int(parts[0][1])
        y = int(parts[1])
        return y, q
    except Exception:
        return None, None


def _next_quarter_label(label: str) -> str:
    """'Q3/2023' -> 'Q4/2023', 'Q4/2023' -> 'Q1/2024'"""
    y, q = _quarter_year(label)
    if y is None:
        return None
    if q < 4:
        return f"Q{q+1}/{y}"
    else:
        return f"Q1/{y+1}"


def compute_forward_returns(rev_df: pd.DataFrame,
                             price_df: pd.DataFrame,
                             growth_threshold: float = 10.0) -> dict:
    """
    Tính forward return sau mỗi quý.
    growth_threshold: ngưỡng tăng trưởng QoQ (%) để xác định signal.
    """
    # Merge
    merged = rev_df.merge(price_df[['Quarter', 'TB_Gia_Ngay']], on='Quarter', how='inner')
    merged['Growth_QoQ_Pct'] = pd.to_numeric(merged['Growth_QoQ_Pct'], errors='coerce')
    merged['TB_Gia_Ngay'] = pd.to_numeric(merged['TB_Gia_Ngay'], errors='coerce')

    # Tính forward return: (giá quý sau / giá quý hiện tại) - 1
    merged['Next_Quarter'] = merged['Quarter'].apply(_next_quarter_label)
    next_price = merged[['Quarter', 'TB_Gia_Ngay']].rename(
        columns={'Quarter': 'Next_Quarter', 'TB_Gia_Ngay': 'Next_Price'}
    )
    merged = merged.merge(next_price, on='Next_Quarter', how='left')
    merged['Forward_Return_Pct'] = (merged['Next_Price'] / merged['TB_Gia_Ngay'] - 1) * 100

    # Signal / Non-Signal
    merged['Signal'] = merged['Growth_QoQ_Pct'] > growth_threshold
    signal_df = merged[merged['Signal'] & merged['Forward_Return_Pct'].notna()]
    no_signal_df = merged[~merged['Signal'] & merged['Forward_Return_Pct'].notna()]
    base_df = merged[merged['Forward_Return_Pct'].notna()]

    # Thống kê
    signal_hits = (signal_df['Forward_Return_Pct'] > 0).sum()
    signal_n = len(signal_df)
    base_hits = (base_df['Forward_Return_Pct'] > 0).sum()
    base_n = len(base_df)

    signal_avg = signal_df['Forward_Return_Pct'].mean()
    base_avg = base_df['Forward_Return_Pct'].mean()
    no_signal_avg = no_signal_df['Forward_Return_Pct'].mean()

    signal_hit_rate = signal_hits / signal_n if signal_n > 0 else np.nan
    base_hit_rate = base_hits / base_n if base_n > 0 else np.nan

    return {
        'detail': merged,
        'signal_n': signal_n,
        'signal_hit_rate': signal_hit_rate,
        'signal_avg_fwd_return_pct': round(signal_avg, 2) if not np.isnan(signal_avg) else np.nan,
        'base_n': base_n,
        'base_hit_rate': base_hit_rate,
        'base_avg_fwd_return_pct': round(base_avg, 2),
        'no_signal_avg_fwd_return_pct': round(no_signal_avg, 2) if not np.isnan(no_signal_avg) else np.nan,
        'alpha_vs_base_pct': round(signal_avg - base_avg, 2) if not np.isnan(signal_avg) else np.nan,
        'threshold': growth_threshold
    }


def run_revenue_signal_backtest(output_dir: str,
                                thresholds: list = [5.0, 10.0, 15.0]) -> dict:
    """Entry point. Chạy backtest tín hiệu doanh thu."""
    print("\n--- BACKTEST: TÍN HIỆU DOANH THU → FORWARD RETURN ---")
    rev_df = _load_revenue(output_dir)
    price_df = _load_price_by_quarter(output_dir)

    all_results = []
    detail_frames = []
    for thresh in thresholds:
        res = compute_forward_returns(rev_df, price_df, growth_threshold=thresh)
        detail = res.pop('detail')
        detail.insert(0, 'Threshold_Pct', thresh)
        detail_frames.append(detail)

        verdict = "✅ Có giá trị" if (
            (res['signal_hit_rate'] or 0) > 0.55 and (res['alpha_vs_base_pct'] or -99) > 2.0
        ) else "⚠️ Yếu"

        print(f"  Ngưỡng QoQ >{thresh:.0f}%: N={res['signal_n']} quý | "
              f"Hit={res['signal_hit_rate']*100:.1f}% | "
              f"Avg Fwd Return={res['signal_avg_fwd_return_pct']:.2f}% | "
              f"Alpha={res['alpha_vs_base_pct']:.2f}% | {verdict}")

        res['Verdict'] = verdict
        res['Threshold_Pct'] = thresh
        all_results.append(res)

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(os.path.join(output_dir, "backtest_revenue_signal.csv"), index=False)

    # Lưu chi tiết quý nào là signal
    if detail_frames:
        detail_all = pd.concat(detail_frames, ignore_index=True)
        detail_all.to_csv(os.path.join(output_dir, "backtest_revenue_detail.csv"), index=False)

    print("  Đã lưu kết quả tín hiệu doanh thu vào output/")
    return {'summary': summary_df, 'details': all_results}
