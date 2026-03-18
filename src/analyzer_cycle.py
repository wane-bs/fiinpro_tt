"""
analyzer_cycle.py
=================
Module phân tích chu kỳ (Common Cycle Analysis).

Mục tiêu:
  - Dùng STL để tách Trend của các chỉ số hiệu quả (ROA, ROE, Biên LN Gộp).
  - Chạy Cross-Correlation giữa các thành phần Trend với các biến cấu trúc
    (Hàng tồn kho, Nợ phải trả, Phải thu) để lập "Bản đồ tác động" theo lag.

Input:  data_dict (từ preprocessor) — dict chứa BCTC, KQKD, LCTT.
Output:
  - output/cycle_decomposition.csv   : Trend/Seasonal/Residual cho ROA, ROE, Biên LN
  - output/cycle_cross_correlation.csv: Ma trận CCF lag (-4..+4 quý)
  - output/cycle_report.md           : Tóm tắt lệch pha chu kỳ
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _get_row(df: pd.DataFrame, col: str, pattern: str) -> pd.Series | None:
    """Trả về Series số liệu của dòng khớp pattern, hoặc None."""
    mask = df[col].str.contains(pattern, case=False, na=False, regex=True)
    if not mask.any():
        return None
    idx = mask.to_numpy().nonzero()[0][0]
    return df.iloc[idx, 1:].astype(float)


def _build_quarterly_series(series: pd.Series) -> pd.Series:
    """
    Chuyển Series với index là label quý (ví dụ 'Q1/2020') thành
    DatetimeIndex dạng period-end để STL có thể nhận diện tần suất.
    Nếu index đã là chuỗi số (column headers) không phải quý, giữ nguyên thứ tự.
    """
    s = series.copy().fillna(0)
    # Thử parse quarter label dạng Q1/2020 hoặc 2020Q1
    try:
        s.index = pd.PeriodIndex(
            [str(x).strip().replace('/', 'Q').replace('Q', 'Q', 1)
             if 'Q' not in str(x) else str(x).strip()
             for x in s.index],
            freq='Q'
        )
        s = s.sort_index()
    except Exception:
        # Fallback: giữ index gốc, không sort
        pass
    return s


def _stl_decompose(series: pd.Series, period: int = 4):
    """
    Thực hiện STL decomposition với statsmodels.
    Trả về (trend, seasonal, resid) Series.
    Fallback về rolling-mean nếu STL không khả dụng.
    """
    values = series.values
    n = len(values)

    # STL yêu cầu tối thiểu 2 × period quan sát
    if n < 2 * period:
        # Fallback: trend = rolling mean window=period
        trend = pd.Series(values).rolling(window=period, center=True, min_periods=1).mean().values
        seasonal = np.zeros(n)
        resid = values - trend
        return (
            pd.Series(trend, index=series.index),
            pd.Series(seasonal, index=series.index),
            pd.Series(resid, index=series.index)
        )

    try:
        from statsmodels.tsa.seasonal import STL
        stl = STL(series, period=period, robust=True)
        result = stl.fit()
        return (
            pd.Series(result.trend, index=series.index),
            pd.Series(result.seasonal, index=series.index),
            pd.Series(result.resid, index=series.index)
        )
    except Exception:
        # Fallback nếu statsmodels không có hoặc lỗi
        trend = pd.Series(values).rolling(window=period, center=True, min_periods=1).mean().values
        seasonal = np.zeros(n)
        resid = values - trend
        return (
            pd.Series(trend, index=series.index),
            pd.Series(seasonal, index=series.index),
            pd.Series(resid, index=series.index)
        )


def _cross_correlation(s1: pd.Series, s2: pd.Series, max_lag: int = 4) -> dict:
    """
    Tính cross-correlation giữa s1 và s2 tại các lag từ -max_lag đến +max_lag.
    Lag dương: s1 dẫn trước s2. Lag âm: s2 dẫn trước s1.
    Chuẩn hóa về [-1, 1].
    Trả về dict {lag: corr_value}.
    """
    # Align on common index
    common = s1.index.intersection(s2.index)
    if len(common) < 5:
        return {lag: np.nan for lag in range(-max_lag, max_lag + 1)}

    a = s1.loc[common].values.astype(float)
    b = s2.loc[common].values.astype(float)

    # Chuẩn hóa
    a = (a - np.nanmean(a))
    b = (b - np.nanmean(b))
    std_a = np.nanstd(a) or 1
    std_b = np.nanstd(b) or 1
    a /= std_a
    b /= std_b

    result = {}
    n = len(a)
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            if n - lag < 2:
                result[lag] = np.nan
            else:
                result[lag] = float(np.corrcoef(a[:n - lag], b[lag:])[0, 1])
        else:
            shift = -lag
            if n - shift < 2:
                result[lag] = np.nan
            else:
                result[lag] = float(np.corrcoef(a[shift:], b[:n - shift])[0, 1])
    return result


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def run_cycle_analysis(data_dict: dict, output_dir: str) -> bool:
    """
    Entry point. Thực hiện STL decomposition + Cross-Correlation.
    Xuất CSV và Markdown report vào output_dir.
    """
    print("\n--- LUỒNG 5b: PHÂN TÍCH CHU KỲ CẤU TRÚC (STL + CCF) ---")

    bctc_df = data_dict.get('Bảng cân đối kế toán')
    kqkd_df = data_dict.get('Kết quả kinh doanh')

    if bctc_df is None or kqkd_df is None:
        print("  ⚠️ Thiếu BCTC hoặc KQKD — bỏ qua Cycle Analysis.")
        return False

    col_b = bctc_df.columns[0]
    col_k = kqkd_df.columns[0]

    # --- 1. Trích xuất các chỉ số hiệu quả ---
    net_income  = _get_row(kqkd_df, col_k, r'Lãi/\(lỗ\) thuần sau thuế|Lợi nhuận sau thuế|Lợi nhuận thuần')
    total_assets = _get_row(bctc_df, col_b, 'TỔNG CỘNG TÀI SẢN')
    total_equity = _get_row(bctc_df, col_b, 'VỐN CHỦ SỞ HỮU')
    gross_profit = _get_row(kqkd_df, col_k, r'Lợi nhuận gộp')
    net_revenue  = _get_row(kqkd_df, col_k, r'Doanh thu thuần|Doanh thu bán hàng')

    performance_series = {}
    if net_income is not None and total_assets is not None:
        roa = (net_income / total_assets.replace(0, np.nan)) * 100
        performance_series['ROA (%)'] = _build_quarterly_series(roa.fillna(0))

    if net_income is not None and total_equity is not None:
        roe = (net_income / total_equity.replace(0, np.nan)) * 100
        performance_series['ROE (%)'] = _build_quarterly_series(roe.fillna(0))

    if gross_profit is not None and net_revenue is not None:
        margin = (gross_profit / net_revenue.replace(0, np.nan)) * 100
        performance_series['Biên LN Gộp (%)'] = _build_quarterly_series(margin.fillna(0))

    if not performance_series:
        print("  ⚠️ Không đủ dữ liệu để tính chỉ số hiệu quả — bỏ qua.")
        return False

    # --- 2. Trích xuất biến cấu trúc (tỷ trọng %) ---
    structural_raw = {}

    # Hàng tồn kho
    htk = _get_row(bctc_df, col_b, r'Hàng tồn kho|HTK')
    if htk is not None and total_assets is not None:
        structural_raw['HTK/TTS (%)'] = _build_quarterly_series(
            (htk / total_assets.replace(0, np.nan) * 100).fillna(0)
        )

    # Nợ phải trả (tổng)
    no_pt = _get_row(bctc_df, col_b, r'NỢ PHẢI TRẢ|Nợ phải trả')
    if no_pt is not None and total_assets is not None:
        structural_raw['Nợ/TTS (%)'] = _build_quarterly_series(
            (no_pt / total_assets.replace(0, np.nan) * 100).fillna(0)
        )

    # Phải thu
    phai_thu = _get_row(bctc_df, col_b, r'Phải thu|phải thu ngắn hạn')
    if phai_thu is None:
        # Thử tên khác
        phai_thu = _get_row(bctc_df, col_b, r'Phải thu khách hàng')
    if phai_thu is not None and total_assets is not None:
        structural_raw['Phải thu/TTS (%)'] = _build_quarterly_series(
            (phai_thu / total_assets.replace(0, np.nan) * 100).fillna(0)
        )

    # Nợ ngắn hạn
    no_nh = _get_row(bctc_df, col_b, r'Vay và nợ thuê tài chính ngắn hạn|Vay ngắn hạn')
    if no_nh is not None and total_assets is not None:
        structural_raw['Nợ NH/TTS (%)'] = _build_quarterly_series(
            (no_nh / total_assets.replace(0, np.nan) * 100).fillna(0)
        )

    # --- 3. STL Decomposition ---
    print("  Đang chạy STL decomposition...")
    decomp_records = []
    trend_series = {}  # Lưu Trend để dùng trong CCF

    for name, s in performance_series.items():
        trend, seasonal, resid = _stl_decompose(s, period=4)
        trend_series[name] = trend
        for i, idx in enumerate(s.index):
            decomp_records.append({
                'Chỉ số': name,
                'Quý': str(idx),
                'Giá trị gốc': round(float(s.iloc[i]), 4),
                'Trend': round(float(trend.iloc[i]), 4),
                'Seasonal': round(float(seasonal.iloc[i]), 4),
                'Residual': round(float(resid.iloc[i]), 4)
            })

    decomp_df = pd.DataFrame(decomp_records)
    decomp_path = os.path.join(output_dir, 'cycle_decomposition.csv')
    decomp_df.to_csv(decomp_path, index=False, encoding='utf-8-sig')
    print(f"  ✓ STL xong — {len(decomp_df)} dòng → {decomp_path}")

    # --- 4. Cross-Correlation ---
    print("  Đang chạy Cross-Correlation (CCF)...")
    ccf_records = []
    best_lag_summary = []  # Cho báo cáo Markdown

    for perf_name, perf_trend in trend_series.items():
        for struct_name, struct_s in structural_raw.items():
            ccf = _cross_correlation(perf_trend, struct_s, max_lag=4)
            for lag, corr in ccf.items():
                ccf_records.append({
                    'Chỉ số Hiệu quả (Trend)': perf_name,
                    'Biến Cấu trúc': struct_name,
                    'Lag (quý)': lag,
                    'Tương quan (CCF)': round(corr, 4) if not np.isnan(corr) else None
                })

            # Tìm lag có |corr| lớn nhất
            valid = {k: v for k, v in ccf.items() if not np.isnan(v)}
            if valid:
                best_lag = max(valid, key=lambda k: abs(valid[k]))
                best_corr = valid[best_lag]
                direction = "dẫn trước" if best_lag > 0 else ("trễ sau" if best_lag < 0 else "đồng thời")
                best_lag_summary.append({
                    'perf': perf_name,
                    'struct': struct_name,
                    'best_lag': best_lag,
                    'best_corr': best_corr,
                    'direction': direction
                })

    ccf_df = pd.DataFrame(ccf_records)
    ccf_path = os.path.join(output_dir, 'cycle_cross_correlation.csv')
    ccf_df.to_csv(ccf_path, index=False, encoding='utf-8-sig')
    print(f"  ✓ CCF xong — {len(ccf_df)} dòng → {ccf_path}")

    # --- 5. Markdown Report ---
    _generate_cycle_report(output_dir, best_lag_summary, len(list(performance_series.values())[0]))
    return True


def _generate_cycle_report(output_dir: str, best_lag_summary: list, n_quarters: int):
    """Tạo cycle_report.md từ kết quả CCF."""
    report_path = os.path.join(output_dir, 'cycle_report.md')

    lines = [
        "# Báo cáo Phân tích Chu kỳ Cấu trúc (STL + Cross-Correlation)\n",
        f"> Phân tích dựa trên {n_quarters} quý dữ liệu. "
        "STL period=4 (chu kỳ quý). Lag dương = biến cấu trúc **dẫn trước** chỉ số hiệu quả.\n\n",
        "## Bản đồ Tác động: Lag có Tương quan Cao nhất\n\n",
        "| Chỉ số Hiệu quả (Trend) | Biến Cấu trúc | Lag (quý) | Tương quan | Diễn giải |\n",
        "|:---|:---|:---:|:---:|:---|\n"
    ]

    for item in sorted(best_lag_summary, key=lambda x: abs(x['best_corr']), reverse=True):
        corr = item['best_corr']
        lag = item['best_lag']
        struct = item['struct']
        perf = item['perf']
        direction = item['direction']

        # Diễn giải chiều tác động
        if lag != 0:
            if lag > 0:
                interp = f"`{struct}` dẫn trước `{perf}` {lag} quý (tương quan {'thuận' if corr > 0 else 'nghịch'})"
            else:
                interp = f"`{perf}` dẫn trước `{struct}` {abs(lag)} quý (tương quan {'thuận' if corr > 0 else 'nghịch'})"
        else:
            interp = f"`{struct}` và `{perf}` biến động đồng thời (lag=0)"

        icon = "🔴" if abs(corr) > 0.5 else ("🟡" if abs(corr) > 0.3 else "⚪")
        lines.append(f"| {perf} | {struct} | {lag:+d} | {icon} {corr:.3f} | {interp} |\n")

    lines.append("\n## Giải thích Ký hiệu\n")
    lines.append("- **Lag > 0:** Biến cấu trúc thay đổi *trước*, sau đó chỉ số hiệu quả phản ứng.\n")
    lines.append("- **Lag < 0:** Chỉ số hiệu quả dẫn trước biến cấu trúc.\n")
    lines.append("- 🔴 |CCF| > 0.5 (tương quan mạnh) | 🟡 |CCF| > 0.3 (trung bình) | ⚪ yếu\n")
    lines.append("\n## Dữ liệu gốc\n")
    lines.append("- **STL Decomposition:** `output/cycle_decomposition.csv`\n")
    lines.append("- **Ma trận CCF đầy đủ:** `output/cycle_cross_correlation.csv`\n")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"  ✓ Đã tạo báo cáo chu kỳ: {report_path}")
