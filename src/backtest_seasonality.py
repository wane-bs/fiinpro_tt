"""
backtest_seasonality.py
=======================
Kiểm định ngược (Walk-Forward) cho giả thuyết mùa vụ giá tháng 4-5.

Phương pháp:
  Với mỗi năm kiểm định Y (từ 2015 đến 2024):
    1. Học tháng nào mạnh nhất trên dữ liệu [đầu mẫu .. 31/12/(Y-1)]
    2. Dự báo: tháng 4-5 trên dữ liệu Y có return trung bình > 0 không?
    3. Kiểm tra so với thực tế
  Tổng hợp Hit Rate, Average Return và Sharpe.

Backtest giao dịch:
  Mua đầu tháng 4, bán cuối tháng 5 — áp dụng 14 năm (2011-2024).
"""

import pandas as pd
import numpy as np
import os


def _load_price_features(output_dir: str) -> pd.DataFrame:
    path = os.path.join(output_dir, "price_analysis_features.csv")
    df = pd.read_csv(path)
    # Parse date
    date_col = next((c for c in df.columns if 'NGÀY' in c.upper() or 'DATE' in c.upper()), None)
    if date_col is None:
        raise ValueError("Không tìm thấy cột ngày trong price_analysis_features.csv")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: 'NGÀY'})
    df = df.sort_values('NGÀY').reset_index(drop=True)
    return df


def walkforward_seasonality_test(price_df: pd.DataFrame,
                                  test_years: range = range(2015, 2025)) -> pd.DataFrame:
    """
    Walk-Forward: với mỗi năm Y, dùng data trước Y để học.
    Dự báo: tháng 4-5 năm Y có avg log return > trung bình không?
    Returns DataFrame kết quả từng năm.
    """
    results = []
    price_df = price_df.copy()
    price_df['YEAR'] = price_df['NGÀY'].dt.year
    price_df['MONTH'] = price_df['NGÀY'].dt.month

    for year in test_years:
        # --- Train: tất cả dữ liệu trước năm Y ---
        train = price_df[price_df['YEAR'] < year]
        if len(train) < 252:  # cần >=1 năm train
            continue

        monthly_train = train.groupby('MONTH')['LOG_RETURN'].mean()
        apr_may_train = monthly_train[[4, 5]].mean() if (4 in monthly_train and 5 in monthly_train) else np.nan
        other_train = monthly_train.drop([4, 5], errors='ignore').mean()

        # Dự báo: APR-MAY sẽ tốt hơn trung bình?
        prediction_positive = (apr_may_train > other_train) if not np.isnan(apr_may_train) else None

        # --- Test: giá thực năm Y tháng 4-5 ---
        test = price_df[(price_df['YEAR'] == year) & (price_df['MONTH'].isin([4, 5]))]
        if len(test) < 10:
            continue

        actual_apr_may = test['LOG_RETURN'].mean()
        actual_others = price_df[
            (price_df['YEAR'] == year) & (~price_df['MONTH'].isin([4, 5]))
        ]['LOG_RETURN'].mean()
        actual_positive = (actual_apr_may > actual_others)

        hit = (prediction_positive == actual_positive) if prediction_positive is not None else None

        results.append({
            'Year': year,
            'Train_AprMay_AvgReturn_Pct': round(apr_may_train * 100, 4) if not np.isnan(apr_may_train) else None,
            'Train_Other_AvgReturn_Pct': round(other_train * 100, 4),
            'Prediction_AprMay_Positive': prediction_positive,
            'Actual_AprMay_AvgReturn_Pct': round(actual_apr_may * 100, 4),
            'Actual_Other_AvgReturn_Pct': round(actual_others * 100, 4),
            'Actual_Positive': actual_positive,
            'Hit': hit
        })

    return pd.DataFrame(results)


def backtest_apr_may_rule(price_df: pd.DataFrame,
                           years: range = range(2011, 2025),
                           fee_pct: float = 0.0015) -> pd.DataFrame:
    """
    Backtest quy tắc: Mua đầu tháng 4, Bán cuối tháng 5.
    Tính Return (%), so với Buy & Hold cùng kỳ.
    fee_pct: phí mỗi lượt (mua + bán = 2 × fee_pct)
    """
    records = []
    price_col = next((c for c in price_df.columns if 'GIÁ' in c.upper() or c.upper() in ['PRICE', 'CLOSE']), None)
    if price_col is None:
        raise ValueError("Không tìm thấy cột giá trong price_analysis_features.csv")

    price_df = price_df.copy()
    price_df['YEAR'] = price_df['NGÀY'].dt.year
    price_df['MONTH'] = price_df['NGÀY'].dt.month

    for year in years:
        apr_may = price_df[(price_df['YEAR'] == year) & (price_df['MONTH'].isin([4, 5]))].copy()
        if len(apr_may) < 5:
            continue

        buy_price = apr_may[price_col].iloc[0]   # đầu tháng 4
        sell_price = apr_may[price_col].iloc[-1]  # cuối tháng 5

        gross_return = (sell_price - buy_price) / buy_price
        net_return = gross_return - 2 * fee_pct  # mua + bán

        # Buy & Hold: từ đầu năm đến cuối năm
        full_year = price_df[price_df['YEAR'] == year]
        bh_return = None
        if len(full_year) >= 2:
            bh_return = (full_year[price_col].iloc[-1] - full_year[price_col].iloc[0]) / full_year[price_col].iloc[0]

        records.append({
            'Year': year,
            'Buy_Price': round(buy_price, 2),
            'Sell_Price': round(sell_price, 2),
            'Gross_Return_Pct': round(gross_return * 100, 2),
            'Net_Return_After_Fee_Pct': round(net_return * 100, 2),
            'BuyHold_Return_Pct': round(bh_return * 100, 2) if bh_return is not None else None,
            'Alpha_vs_BuyHold_Pct': round((net_return - bh_return) * 100, 2) if bh_return is not None else None,
            'Win': net_return > 0
        })

    return pd.DataFrame(records)


def compute_sharpe(returns_series: pd.Series, risk_free: float = 0.045) -> float:
    """Tính Sharpe Ratio annualized từ chuỗi return (decimal)."""
    if len(returns_series) < 2:
        return np.nan
    # Coi mỗi window ~2 tháng → ~6 windows/năm
    excess = returns_series - (risk_free / 6)
    return round(excess.mean() / excess.std() * np.sqrt(6), 3)


def run_seasonality_backtest(output_dir: str) -> dict:
    """
    Entry point. Chạy toàn bộ kiểm định mùa vụ.
    Returns dict với DataFrames kết quả.
    """
    print("\n--- BACKTEST: KIỂM ĐỊNH MÙA VỤ GIÁ THÁNG 4-5 ---")
    price_df = _load_price_features(output_dir)

    # 1. Walk-Forward Test
    wf_df = walkforward_seasonality_test(price_df)
    hit_rate = wf_df['Hit'].sum() / len(wf_df) if len(wf_df) > 0 else 0
    print(f"  Walk-Forward Hit Rate: {hit_rate*100:.1f}% ({int(wf_df['Hit'].sum())}/{len(wf_df)} năm)")

    # 2. Backtest giao dịch (không phí)
    trade_df_nofee = backtest_apr_may_rule(price_df, fee_pct=0.0)
    trade_df_fee = backtest_apr_may_rule(price_df, fee_pct=0.0015)

    win_rate = trade_df_fee['Win'].mean() if len(trade_df_fee) > 0 else 0
    avg_net_return = trade_df_fee['Net_Return_After_Fee_Pct'].mean()
    sharpe = compute_sharpe(trade_df_fee['Net_Return_After_Fee_Pct'] / 100)
    avg_alpha = trade_df_fee['Alpha_vs_BuyHold_Pct'].mean()

    print(f"  Win Rate (có phí): {win_rate*100:.1f}%")
    print(f"  Avg Net Return T4-T5: {avg_net_return:.2f}%")
    print(f"  Avg Alpha vs Buy&Hold: {avg_alpha:.2f}%")
    print(f"  Sharpe Ratio: {sharpe:.3f}")

    # Lưu CSV
    wf_df.to_csv(os.path.join(output_dir, "backtest_walkforward_seasonality.csv"), index=False)
    trade_df_fee.to_csv(os.path.join(output_dir, "backtest_apr_may_trading.csv"), index=False)
    print(f"  Đã lưu kết quả vào output/")

    return {
        'walkforward': wf_df,
        'trade_fee': trade_df_fee,
        'hit_rate': hit_rate,
        'win_rate': win_rate,
        'avg_net_return': avg_net_return,
        'sharpe': sharpe,
        'avg_alpha': avg_alpha
    }
