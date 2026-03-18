"""
analyzer_valuation.py
=====================
Tru cot 3: Valuation Framework — Revenue-based Target Price

3 phuong phap tinh Target Price:
  A. P/S Multiple  : PS_hist x Forward Revenue
  B. Log-Linear Reg: log(Price) ~ log(Rev) + QoQ + sin/cos seasonality
  C. Mean Reversion: Rolling 8Q Fair Value +/- 1.5*Volatility Band

Output:
  output/valuation_bands.csv       — Fair Value & Bands theo quy
  output/target_price_scenarios.csv — Bear / Base / Bull target hien tai
"""

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# ── Helpers ──────────────────────────────────────────────────────────

def _quarter_to_int(label: str) -> int:
    """'Q2/2023' -> 2023*4 + 1  (0-indexed quarter)."""
    try:
        q = int(label.strip()[1])
        y = int(label.strip().split('/')[1])
        return y * 4 + (q - 1)
    except Exception:
        return 0


def _load_merged(output_dir: str) -> pd.DataFrame:
    """Doc va merge aggregated price + revenue growth."""
    rev = pd.read_csv(os.path.join(output_dir, 'revenue_growth.csv'))
    rev.columns = [c.strip() for c in rev.columns]
    rev = rev.rename(columns={'Quý': 'Quarter', 'Doanh_Thu': 'Revenue',
                              'Tang_Truong_QoQ_Pct': 'Rev_QoQ'})

    price = pd.read_csv(os.path.join(output_dir, 'aggregated_price_by_quarter.csv'))
    price.columns = [c.strip() for c in price.columns]
    price = price.rename(columns={'BCTC_Quarter_Label': 'Quarter',
                                  'TB_Gia_Ngay': 'Price',
                                  'TB_KhoiLuong_KhopLenh_Ngay': 'Volume',
                                  'TB_BienDong': 'Volatility'})

    merged = rev.merge(price[['Quarter', 'Price', 'Volume', 'Volatility']],
                       on='Quarter', how='inner')
    merged['sort_key'] = merged['Quarter'].apply(_quarter_to_int)
    merged = merged.sort_values('sort_key').reset_index(drop=True)
    merged['Revenue'] = pd.to_numeric(merged['Revenue'], errors='coerce')
    merged['Price'] = pd.to_numeric(merged['Price'], errors='coerce')
    merged['Rev_QoQ'] = pd.to_numeric(merged['Rev_QoQ'], errors='coerce')
    merged['Volatility'] = pd.to_numeric(merged['Volatility'], errors='coerce')

    return merged.dropna(subset=['Revenue', 'Price'])


# ── Phuong phap A: P/S Multiple ──────────────────────────────────────

def method_a_ps_multiple(df: pd.DataFrame,
                           lookback_q: int = 16,
                           forward_cagr_pct: float = None) -> dict:
    """
    PS_hist = median(Price / Revenue) qua lookback_q quy gan nhat.
    Forward Revenue = Rev_last * (1 + CAGR)^1 (1 nam toi).
    """
    recent = df.tail(lookback_q).copy()
    ps_ratios = recent['Price'] / recent['Revenue']
    ps_hist_median = ps_ratios.median()
    ps_hist_mean = ps_ratios.mean()

    rev_last = df['Revenue'].iloc[-1]

    # Uoc tinh CAGR tu chính data (12 quy)
    if forward_cagr_pct is None:
        if len(df) >= 12:
            rev_12q_ago = df['Revenue'].iloc[-12]
            if rev_12q_ago > 0:
                forward_cagr_pct = ((rev_last / rev_12q_ago) ** (1 / 3) - 1) * 100
            else:
                forward_cagr_pct = 10.0  # fallback
        else:
            forward_cagr_pct = 10.0

    cagr = forward_cagr_pct / 100
    rev_forward = rev_last * (1 + cagr)

    target_base = ps_hist_median * rev_forward
    target_bull = ps_hist_median * rev_forward * 1.15  # +15% premium
    target_bear = ps_hist_median * rev_forward * 0.85  # -15% discount

    return {
        'method': 'A_PS_Multiple',
        'PS_hist_median': round(ps_hist_median, 6),
        'PS_hist_mean': round(ps_hist_mean, 6),
        'forward_CAGR_pct': round(forward_cagr_pct, 2),
        'rev_last': round(rev_last, 2),
        'rev_forward_est': round(rev_forward, 2),
        'target_bear': round(target_bear, 2),
        'target_base': round(target_base, 2),
        'target_bull': round(target_bull, 2),
    }


# ── Phuong phap B: Log-Linear Regression ─────────────────────────────

def method_b_log_regression(df: pd.DataFrame) -> dict:
    """
    log(Price) ~ log(Revenue) + Rev_QoQ + sin(season) + cos(season)
    Su dung Ridge regression. Train tren 80%, kiem tra tren 20%.
    """
    df2 = df.copy().dropna(subset=['Revenue', 'Price', 'Rev_QoQ'])
    df2 = df2[df2['Revenue'] > 0][df2['Price'] > 0]

    # Features
    df2['log_rev'] = np.log(df2['Revenue'])
    df2['log_price'] = np.log(df2['Price'])
    df2['qoy_norm'] = df2['Rev_QoQ'].fillna(0)

    # Seasonality encode tu quarter index (sin/cos)
    df2['q_num'] = df2['sort_key'] % 4  # 0=Q1 .. 3=Q4
    df2['season_sin'] = np.sin(2 * np.pi * df2['q_num'] / 4)
    df2['season_cos'] = np.cos(2 * np.pi * df2['q_num'] / 4)

    feature_cols = ['log_rev', 'qoy_norm', 'season_sin', 'season_cos']
    X = df2[feature_cols].values
    y = df2['log_price'].values

    # Train/test (80/20 theo thoi gian)
    split = int(len(X) * 0.80)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_s, y_train)

    y_pred_test = model.predict(X_test_s)
    r2_oos = 1 - np.sum((y_test - y_pred_test) ** 2) / np.sum((y_test - y_test.mean()) ** 2)

    # Du bao Q tiep theo (CAGR tu phuong phap A)
    last_row = df2.iloc[-1]
    cagr_4q = 0.10  # fallback
    if len(df2) >= 12:
        rev_12q = df2['Revenue'].iloc[-12]
        if rev_12q > 0:
            cagr_4q = (last_row['Revenue'] / rev_12q) ** (1 / 3) - 1

    rev_next = last_row['Revenue'] * (1 + cagr_4q)
    qoy_next = cagr_4q * 4 * 100  # uoc tinh QoQ quy toi

    q_next = (last_row['sort_key'] + 1) % 4
    feat_next = np.array([[np.log(rev_next), qoy_next,
                           np.sin(2 * np.pi * q_next / 4),
                           np.cos(2 * np.pi * q_next / 4)]])
    feat_next_s = scaler.transform(feat_next)
    log_target = model.predict(feat_next_s)[0]
    target_base = np.exp(log_target)

    price_last = last_row['Price']
    resid_std = np.std(y_train - model.predict(scaler.transform(X_train)))
    target_bull = np.exp(log_target + 1.5 * resid_std)
    target_bear = np.exp(log_target - 1.5 * resid_std)

    return {
        'method': 'B_LogLinear_Regression',
        'R2_OOS': round(r2_oos, 4),
        'target_bear': round(target_bear, 2),
        'target_base': round(target_base, 2),
        'target_bull': round(target_bull, 2),
        'price_last': round(price_last, 2),
    }


# ── Phuong phap C: Mean Reversion Band ───────────────────────────────

def method_c_mean_reversion(df: pd.DataFrame, window_q: int = 8) -> pd.DataFrame:
    """
    Fair_Value = rolling window_q avg Price
    Upper = FV * (1 + 1.5 * Volatility)
    Lower = FV * (1 - 1.5 * Volatility)
    """
    df2 = df.copy()
    df2['Fair_Value'] = df2['Price'].rolling(window_q).mean()
    vol = df2['Volatility'].fillna(df2['Volatility'].median())
    df2['Upper_Band'] = df2['Fair_Value'] * (1 + 1.5 * vol)
    df2['Lower_Band'] = df2['Fair_Value'] * (1 - 1.5 * vol)

    # Vi tri gia hien tai trong band
    last = df2.iloc[-1]
    fv = last['Fair_Value']
    ub = last['Upper_Band']
    lb = last['Lower_Band']
    price_now = last['Price']

    if not pd.isna(fv) and ub > lb:
        position = (price_now - lb) / (ub - lb)  # 0=day, 1=dinh
    else:
        position = np.nan

    df2['Band_Position'] = np.where(
        (df2['Upper_Band'] - df2['Lower_Band']) > 0,
        (df2['Price'] - df2['Lower_Band']) / (df2['Upper_Band'] - df2['Lower_Band']),
        np.nan
    )

    return df2[['Quarter', 'Price', 'Fair_Value', 'Upper_Band', 'Lower_Band', 'Band_Position']]


# ── Entry Point ───────────────────────────────────────────────────────

def run_valuation_analysis(output_dir: str) -> dict:
    """
    Doc CSV tu output_dir, chay ca 3 phuong phap, luu ket qua.
    """
    print("\n--- LUONG 6: VALUATION FRAMEWORK ---")
    df = _load_merged(output_dir)
    price_now = df['Price'].iloc[-1]
    print(f"  Gia hien tai (quy cuoi): {price_now:,.0f}")

    # -- Method A --
    res_a = method_a_ps_multiple(df)
    print(f"  [A] P/S Multiple  -> Bear:{res_a['target_bear']:,.0f} | "
          f"Base:{res_a['target_base']:,.0f} | Bull:{res_a['target_bull']:,.0f}")

    # -- Method B --
    res_b = method_b_log_regression(df)
    print(f"  [B] Log-Reg (R2OOS={res_b['R2_OOS']:.3f}) -> "
          f"Bear:{res_b['target_bear']:,.0f} | Base:{res_b['target_base']:,.0f} | Bull:{res_b['target_bull']:,.0f}")

    # -- Method C --
    bands_df = method_c_mean_reversion(df)
    last_band = bands_df.iloc[-1]
    fv = last_band['Fair_Value']
    ub = last_band['Upper_Band']
    lb = last_band['Lower_Band']
    pos = last_band['Band_Position']
    print(f"  [C] Mean Reversion -> Lower:{lb:,.0f} | FairValue:{fv:,.0f} | Upper:{ub:,.0f} "
          f"| Vi tri hien tai: {pos:.1%}" if not pd.isna(pos) else "  [C] Mean Reversion -> N/A")

    # -- Tong hop Target Price (trung binh 3 phuong phap) --
    targets = {
        'bear': np.nanmean([res_a['target_bear'], res_b['target_bear'], lb]),
        'base': np.nanmean([res_a['target_base'], res_b['target_base'], fv]),
        'bull': np.nanmean([res_a['target_bull'], res_b['target_bull'], ub]),
    }

    scenarios_df = pd.DataFrame([
        {'Scenario': 'Bear', 'Method_A': res_a['target_bear'], 'Method_B': res_b['target_bear'],
         'Method_C': round(lb, 2), 'Avg_Target': round(targets['bear'], 2),
         'Upside_Pct': round((targets['bear'] / price_now - 1) * 100, 2)},
        {'Scenario': 'Base', 'Method_A': res_a['target_base'], 'Method_B': res_b['target_base'],
         'Method_C': round(fv, 2), 'Avg_Target': round(targets['base'], 2),
         'Upside_Pct': round((targets['base'] / price_now - 1) * 100, 2)},
        {'Scenario': 'Bull', 'Method_A': res_a['target_bull'], 'Method_B': res_b['target_bull'],
         'Method_C': round(ub, 2), 'Avg_Target': round(targets['bull'], 2),
         'Upside_Pct': round((targets['bull'] / price_now - 1) * 100, 2)},
    ])

    # Luu files
    bands_df.to_csv(os.path.join(output_dir, 'valuation_bands.csv'), index=False)
    scenarios_df.to_csv(os.path.join(output_dir, 'target_price_scenarios.csv'), index=False)

    print(f"\n  === TARGET PRICE TONG HOP ===")
    for _, row in scenarios_df.iterrows():
        icon = "📈" if row['Upside_Pct'] > 0 else "📉"
        print(f"  {icon} {row['Scenario']:4s}: {row['Avg_Target']:>10,.0f} VND "
              f"({row['Upside_Pct']:+.1f}% vs hien tai)")

    print("  Da luu: valuation_bands.csv, target_price_scenarios.csv")

    return {
        'method_a': res_a,
        'method_b': res_b,
        'bands': bands_df,
        'scenarios': scenarios_df,
        'band_position': float(pos) if not pd.isna(pos) else None,
        'price_now': price_now,
        'targets': targets
    }


# ── Module 3.3: Multi-Multiple Valuation ─────────────────────────────

def _find_row_chi_so(df: pd.DataFrame, keyword: str, col_idx: int = 0):
    """Tìm dòng chứa keyword trong sheet chỉ số."""
    col = df.columns[col_idx]
    mask = df[col].str.contains(keyword, case=False, na=False, regex=True)
    if not mask.any():
        return None
    return df.loc[mask.idxmax(), df.columns[1:]].astype(float)


def run_multiples_valuation(data_dict: dict, output_dir: str) -> pd.DataFrame:
    """
    Module 3.3 — Multi-Multiple Valuation.
    Extract P/E, P/B, EV/EBITDA, P/S from chỉ số sheet.
    Output: output/multiples_valuation.csv
    """
    print("\n--- MODULE 3.3: MULTI-MULTIPLE VALUATION ---")

    chi_so = data_dict.get('chỉ số')
    if chi_so is None:
        print("  [WARN] Thiếu sheet 'chỉ số' — bỏ qua Multi-Multiple.")
        return pd.DataFrame()

    quarters = chi_so.columns[1:].tolist()

    # Extract ratios from chỉ số sheet
    pe = _find_row_chi_so(chi_so, 'P/E cơ bản')
    pb = _find_row_chi_so(chi_so, 'P/B')
    ps = _find_row_chi_so(chi_so, 'P/S')
    ev_ebitda = _find_row_chi_so(chi_so, 'doanh nghiệp/ EBITDA|EV/EBITDA')
    ev_ebit = _find_row_chi_so(chi_so, 'doanh nghiệp/ EBIT')
    eps = _find_row_chi_so(chi_so, 'EPS cơ bản')
    bvps = _find_row_chi_so(chi_so, 'BVPS')
    market_cap = _find_row_chi_so(chi_so, 'Vốn hóa')

    result = pd.DataFrame({'Quarter': quarters})

    # Add available metrics
    if pe is not None:
        result['PE_Ratio'] = pe.values
    if pb is not None:
        result['PB_Ratio'] = pb.values
    if ps is not None:
        result['PS_Ratio'] = ps.values
    if ev_ebitda is not None:
        result['EV_EBITDA'] = ev_ebitda.values
    if ev_ebit is not None:
        result['EV_EBIT'] = ev_ebit.values
    if eps is not None:
        result['EPS'] = eps.values
    if bvps is not None:
        result['BVPS'] = bvps.values
    if market_cap is not None:
        result['Market_Cap'] = market_cap.values

    # Round
    numeric_cols = result.select_dtypes(include=np.number).columns
    result[numeric_cols] = result[numeric_cols].round(4)

    out_path = os.path.join(output_dir, 'multiples_valuation.csv')
    result.to_csv(out_path, index=False)

    # Print summary
    last = result.iloc[-1]
    print(f"  [Latest Quarter: {last['Quarter']}]")
    for col in ['PE_Ratio', 'PB_Ratio', 'PS_Ratio', 'EV_EBITDA', 'EPS', 'BVPS']:
        val = last.get(col, np.nan)
        if not pd.isna(val):
            print(f"    {col}: {val:.2f}")
    print(f"  Đã lưu: {out_path}")

    return result

