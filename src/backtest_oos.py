"""
backtest_oos.py
===============
Out-of-Sample (OOS) Validation cho các mô hình RandomForest.

Thiết kế:
  Train set: Q1/2010 → Q4/2021  (48 quý đầu)
  Test set:  Q1/2022 → Q4/2025  (16 quý cuối — ~25% dữ liệu)

Kiểm tra R², RMSE, MAPE trên Test set để phát hiện overfitting.
Ngưỡng chấp nhận:
  - R² OOS ≥ 0.90
  - MAPE ≤ 5% (Tài sản, NV) hoặc ≤ 8% (Doanh thu)
  - RMSE_OOS / RMSE_IS ≤ 2.0
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


# ── Hằng số phân chia ───────────────────────────────────────────────
TRAIN_CUTOFF = "Q4/2021"   # Splits up to and including này là TRAIN
# ─────────────────────────────────────────────────────────────────────


def _quarter_label_to_sort_key(label: str) -> int:
    """Chuyển 'Q1/2022' → số nguyên để sort (2022*4 + 0)."""
    try:
        parts = label.strip().split('/')
        q = int(parts[0][1])  # 'Q1' → 1
        y = int(parts[1])
        return y * 4 + (q - 1)
    except Exception:
        return 0


def _load_timeseries(output_dir: str) -> pd.DataFrame:
    path = os.path.join(output_dir, "structure_timeseries.csv")
    df = pd.read_csv(path)
    df['sort_key'] = df['Quarter'].apply(_quarter_label_to_sort_key)
    return df.sort_values('sort_key').reset_index(drop=True)


def _split_train_test(X: pd.DataFrame, y: pd.Series):
    """
    Tách train/test theo cutoff Q4/2021.
    X.index là các nhãn quý (e.g. 'Q1/2022').
    """
    cutoff_key = _quarter_label_to_sort_key(TRAIN_CUTOFF)
    keys = pd.Series(X.index).apply(_quarter_label_to_sort_key).values
    train_mask = keys <= cutoff_key
    test_mask = ~train_mask

    X_train = X.iloc[train_mask]
    X_test = X.iloc[test_mask]
    y_train = y.iloc[train_mask].astype(float)
    y_test = y.iloc[test_mask].astype(float)
    return X_train, X_test, y_train, y_test


def _mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def run_oos_for_target(ts_df: pd.DataFrame, target_name: str) -> dict:
    """Chạy OOS validation cho 1 target."""
    sub = ts_df[ts_df['Target_Name'] == target_name].copy()

    # Lấy target series
    target_rows = sub[sub['Variable_Type'] == 'Target'][['Quarter', 'Value']].drop_duplicates('Quarter')
    target_rows = target_rows.set_index('Quarter')['Value']

    # Lấy features (pivot)
    feat_rows = sub[sub['Variable_Type'] == 'Feature']
    if feat_rows.empty:
        return {'target': target_name, 'error': 'No features'}

    feat_pivot = feat_rows.pivot_table(index='Quarter', columns='Variable_Name', values='Value', aggfunc='first')
    feat_pivot = feat_pivot.fillna(0)

    # Align
    common_idx = feat_pivot.index.intersection(target_rows.index)
    X = feat_pivot.loc[common_idx].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = pd.to_numeric(target_rows.loc[common_idx], errors='coerce').fillna(0)

    if len(X) < 20:
        return {'target': target_name, 'error': f'Quá ít mẫu: {len(X)}'}

    X_train, X_test, y_train, y_test = _split_train_test(X, y)

    if len(X_train) < 10 or len(X_test) < 4:
        return {'target': target_name, 'error': 'Train hoặc Test set quá nhỏ'}

    # Huấn luyện
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # In-sample
    y_pred_is = model.predict(X_train)
    r2_is = r2_score(y_train, y_pred_is)
    rmse_is = np.sqrt(mean_squared_error(y_train, y_pred_is))
    mape_is = _mape(y_train, y_pred_is)

    # Out-of-sample
    y_pred_oos = model.predict(X_test)
    r2_oos = r2_score(y_test, y_pred_oos)
    rmse_oos = np.sqrt(mean_squared_error(y_test, y_pred_oos))
    mape_oos = _mape(y_test, y_pred_oos)

    overfit_ratio = rmse_oos / rmse_is if rmse_is > 0 else np.nan

    # Verdict
    thresholds = {
        'Tổng Tài sản':   {'r2_min': 0.90, 'mape_max': 5.0},
        'Tổng Nguồn vốn': {'r2_min': 0.90, 'mape_max': 5.0},
        'Tổng Doanh thu': {'r2_min': 0.85, 'mape_max': 8.0},
    }
    th = thresholds.get(target_name, {'r2_min': 0.85, 'mape_max': 8.0})
    pass_r2 = r2_oos >= th['r2_min']
    pass_mape = mape_oos <= th['mape_max']
    pass_ratio = overfit_ratio <= 2.0
    verdict = "✅ ĐẠT" if (pass_r2 and pass_mape and pass_ratio) else "⚠️ CẦN XEM XÉT"

    # Lưu predictions
    pred_df = pd.DataFrame({
        'Quarter': X_test.index,
        'y_true': y_test.values,
        'y_pred_oos': y_pred_oos,
        'Error_Abs': np.abs(y_test.values - y_pred_oos),
        'Error_Pct': np.abs((y_test.values - y_pred_oos) / (y_test.values + 1e-9)) * 100
    })

    return {
        'target': target_name,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'R2_IS': round(r2_is, 4),
        'RMSE_IS': round(rmse_is, 2),
        'MAPE_IS_Pct': round(mape_is, 2),
        'R2_OOS': round(r2_oos, 4),
        'RMSE_OOS': round(rmse_oos, 2),
        'MAPE_OOS_Pct': round(mape_oos, 2),
        'RMSE_OOS_vs_IS_Ratio': round(overfit_ratio, 3),
        'Pass_R2': pass_r2,
        'Pass_MAPE': pass_mape,
        'Pass_Ratio': pass_ratio,
        'Verdict': verdict,
        'predictions': pred_df
    }


def run_oos_validation(output_dir: str) -> dict:
    """Entry point. Chạy OOS cho tất cả các targets."""
    print("\n--- BACKTEST: OOS VALIDATION RANDOMFOREST ---")
    print(f"  Train cutoff: {TRAIN_CUTOFF} | Test: Q1/2022 → Q4/2025")

    ts_df = _load_timeseries(output_dir)
    targets = ts_df['Target_Name'].unique().tolist()

    summary = []
    pred_frames = []
    for target in targets:
        print(f"  Kiểm định OOS: {target}...")
        result = run_oos_for_target(ts_df, target)
        pred_df = result.pop('predictions', pd.DataFrame())
        if not pred_df.empty:
            pred_df.insert(0, 'Target', target)
            pred_frames.append(pred_df)
        summary.append(result)
        if 'error' in result:
            print(f"    ⚠️ {result['error']}")
        else:
            print(f"    R²IS={result['R2_IS']:.4f} | R²OOS={result['R2_OOS']:.4f} | "
                  f"MAPE_OOS={result['MAPE_OOS_Pct']:.2f}% | RMSE Ratio={result['RMSE_OOS_vs_IS_Ratio']:.2f} | "
                  f"{result['Verdict']}")

    summary_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'predictions'} for r in summary])
    summary_df.to_csv(os.path.join(output_dir, "backtest_oos_summary.csv"), index=False)

    if pred_frames:
        all_preds = pd.concat(pred_frames, ignore_index=True)
        all_preds.to_csv(os.path.join(output_dir, "backtest_oos_predictions.csv"), index=False)

    print("  Đã lưu kết quả OOS vào output/")
    return {'summary': summary_df, 'details': summary}
