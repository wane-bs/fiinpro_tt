# Technical Specification — Fiinpro Analysis Pipeline (v2.0)

Tài liệu diễn giải kiến trúc, logic tính toán và Data Flow của dự án sau khi tái cấu trúc sang triết lý **Phân tích Cơ cấu (Structural Analysis)** thay vì Dự báo (Prediction).

> **Nguyên tắc bất biến:**
> 1. Không dùng random split — bắt buộc Time-Series Split
> 2. Không trộn biến giữa các báo cáo ở giai đoạn Vertical Analysis
> 3. OOS R² = thước đo **độ ổn định trọng số**, không phải sức mạnh dự báo
> 4. Common-size (%) là bắt buộc trước khi đưa vào ML

---

## 1. Tổng quan Kiến trúc

```
data.xlsx (6 sheets)
    ↓ data_loader.py
    ↓ preprocessor.py
    ↓
    ┌─────────────────────────────────────────────┐
    │  Chương 1  │  Chương 2  │  Chương 3  │ Ch4 │
    │  Nền tảng  │  Hiệu suất │  Định giá  │Cấu  │
    │  Tĩnh      │  + Chu kỳ  │  + Độ nhạy │trúc │
    └─────────────────────────────────────────────┘
                         ↓
                  signal_engine.py
                         ↓
                 Composite Score / Recommendation
                         ↓
                   app.py (Streamlit)
                   5 Tab theo Chương
```

---

## 2. Chi tiết từng Module

### Bước 0: Tiền xử lý

**`data_loader.py` — `load_raw_with_audit()`**
- Đọc 6 sheets từ `data.xlsx`
- Bóc tách dòng 0 (trạng thái kiểm toán) → `audit_dict` riêng biệt
- Trả về `(raw_data_dict, audit_dict)`

**`preprocessor.py`**
- Fill NaN = 0; chuẩn hóa kiểu cột
- Đánh dấu outlier khối lượng (Percentile 99%) → cờ giao dịch thỏa thuận bất thường

---

### Bước 1 — Chương 1: Nền tảng Tĩnh

**`analyzer_preanalysis.py`**
- Tính `audit_rate_pct = audited_Q / total_Q × 100`
- Tìm chuỗi quý chưa kiểm toán liên tiếp dài nhất
- Quét sheet "Thuyết minh" bằng Regex tìm keyword chính sách kế toán
- Output: `preanalysis_report.json`

**`analyzer_ratios.py` — Vertical Analysis (Module 1.3)**
- BCĐKT: `tỷ trọng = khoản mục / TỔNG CỘNG TÀI SẢN × 100`
- KQKD: `tỷ trọng = khoản mục / Doanh thu thuần × 100`
- Output: `vertical_analysis.csv`

**`analyzer_ratios.py` — Solvency & Liquidity (Module 1.4)**
```
Current Ratio     = TSNH / NNH
Quick Ratio       = (TSNH - HTK) / NNH
D/E Ratio         = Tổng Nợ / Vốn CSH
Debt Ratio        = Tổng Nợ / Tổng TS
Interest Coverage = EBIT / |Lãi vay|
```
- Output: bổ sung vào `financial_ratios.csv`

---

### Bước 2 — Chương 2: Hiệu suất Động

**`analyzer_ratios.py` — Activity & Profitability (Module 2.2)**
```
Asset Turnover     = Doanh thu / Avg(Tổng TS)
Inventory Turnover = GVHB / Avg(HTK)
ROA                = Net Income / Avg(Tổng TS) × 100
ROE                = Net Income / Avg(Vốn CSH)  × 100
```
- Dùng trung bình đầu/cuối kỳ (rolling 2 quý liền kề)

**`analyzer_dupont.py` — DuPont 3 Nhân tố (Module 2.3)**
```
ROE = Net Margin × Asset Turnover × Equity Multiplier
    = (NI/Rev)  × (Rev/AvgAssets) × (AvgAssets/AvgEquity)
```
- Delta ROE = ROE_t − ROE_{t-1} → Waterfall chart
- Output: `dupont_analysis.csv`

**`analyzer_cashflow.py` — Cash Flow Quality (Module 2.4)**
```
CFO/NI Ratio  = CFO / Net Income          (ngưỡng cảnh báo: < 0.7 liên tiếp 3Q)
FCF           = CFO − |Capex|
Accrual Ratio = (NI − CFO) / Avg Assets   (ngưỡng cảnh báo: > 5%)
```
- Cờ cảnh báo tự động gắn vào cột `Flag`
- Output: `cashflow_quality.csv`

---

### Bước 3 — Chương 3: Định giá Tương lai

**`analyzer_dcf.py` — DCF + Sensitivity (Module 3.2)**

Bước 1 — FCFF lịch sử:
```
FCFF = EBIT × (1 - Tax) + Khấu hao − ΔWorking Capital − Capex
Tax = 20% (mặc định VN)
```

Bước 2 — WACC từ dữ liệu nội tại:
```
r_e  = r_f + β × ERP  (CAPM: r_f=4.5%, ERP=7%, β=1.0)
r_d  = |Chi phí lãi vay| / Tổng Nợ × 4 (annualize)
WACC = E/V × r_e + D/V × r_d × (1 - Tax)
```

Bước 3 — DCF 5 năm + Terminal Value:
```
PV = Σ FCFF_base × (1+g)^t / (1+WACC)^t   [t=1..5]
TV = FCFF_5 × (1+g_tv) / (WACC - g_tv) / (1+WACC)^5
Equity Value = PV + TV - Net Debt
```

Bước 4 — Ma trận độ nhạy: WACC ± 2% (bước 0.5%) × g ± 2% (bước 0.5%)
- Output: `dcf_valuation.csv`, `dcf_sensitivity.csv`

**`analyzer_valuation.py` — Valuation Bands + Multiples (Module 3.3)**
- Method A: P/S Multiple × Forward Revenue (CAGR lịch sử)
- Method B: Log-Linear Regression `log(Price) ~ log(Rev) + QoQ + sin/cos(season)`
- Method C: Mean Reversion `Fair Value = rolling 8Q avg; Band = FV ± 1.5σ`
- Tổng hợp Bear/Base/Bull = trung bình 3 phương pháp
- Output: `valuation_bands.csv`, `target_price_scenarios.csv`, `multiples_valuation.csv`

---

### Bước 4 — Chương 4: Cấu trúc & Chu kỳ

**`analyzer_cycle.py` — STL + CCF (Module 4a)**

STL Decomposition (Seasonal-Trend using Loess):
```
Y = Trend + Seasonal + Remainder   [period=4 quý]
Fallback: Rolling mean nếu N < 8
```

Cross-Correlation Function (Lag −4 → +4 quý):
```
r(lag) = corr(Trend_Y[0:n-lag], Struct_X[lag:n])
Lag > 0: biến cấu trúc dẫn trước chỉ số hiệu quả
```
- Output: `cycle_decomposition.csv`, `cycle_cross_correlation.csv`, `cycle_report.md`

**`analyzer_structure.py` — Dual-Auditor (Module 4b)**

Bài toán: $P \gg N$ → cần 2 chiến lược đối lập:

| | ElasticNet (Regularization) | PLSR (Dimension Reduction) |
|:---|:---|:---|
| Mục tiêu | Ép hệ số biến rác về 0 | Nén P biến → k latent factors |
| Output | Mean Coefficient (có dấu +/−) | Mean VIP Score (tầm quan trọng) |
| Ý nghĩa | Chiều hướng tác động | Tầm quan trọng tổng thể |

Training: TimeSeriesSplit (n_splits=5, không random)
Features: Common-size (%) — bắt buộc trước khi fit
Target: ROA (mảng Tài sản), ROE (mảng Nguồn vốn), Biên LN Gộp (KQKD)

**VIP Score (Variable Importance in Projection):**
```python
vip[i] = sqrt(P × Σ_j [s_j × (w[i,j]/‖w[:,j]‖)²] / total_s)
```

Giao thức đọc kết quả 2 lớp:
- **Đồng thuận:** `VIP > 1.0` + `|Coef| >> 0.05` → Key Structural Driver
- **Phân kỳ:** `VIP > 1.0` + `Coef ≈ 0` → Group Effect / Đa cộng tuyến khu vực
- Output: `structure_impact.csv`, `model_metrics.csv`, `analysis_report.md`

---

### Bước 5 — Tổng hợp Tín hiệu

**`signal_engine.py`**
- Tổng hợp điểm: Momentum giá + Định giá P/S + Mùa vụ Doanh thu + Sức khỏe tài chính
- Composite Score (thang −100 đến +100):
  - `> +40`: MUA MẠNH | `+15~+40`: MUA | `−15~+15`: TRUNG LẬP | `< −15`: BÁN
- Output: `composite_signal.csv`, `recommendation.json`

---

### Bước 6 — Dashboard

**`app.py` — Streamlit (5 tab)**

| Tab | Nội dung |
|:---|:---|
| 📘 Chương 1 | Audit Gauge + Common-size Stacked Bar |
| 📗 Chương 2 | DuPont Waterfall + Accrual Dual-Axis + Cycle Z-score Overlay |
| 📙 Chương 3 | Valuation Bands + DCF Heatmap + Target Scenarios |
| 📕 Chương 4 | STL Plot + CCF Heatmap + Factor Importance (VIP + Coef) |
| 📋 Báo cáo | Composite Score Gauge + Phương pháp luận + Markdown reports |

---

## 3. Xử lý Dữ liệu Mất Mát

- `_find_row(df, keyword)` dùng Regex case-insensitive → chịu được typo của FiinPro export
- `np.where(denominator != 0, numerator/denominator, np.nan)` → tránh DivideByZero
- STL: `min_periods=1` + fallback rolling mean nếu N < 2×period
- ElasticNetCV: tự chọn `alpha` và `l1_ratio` tối ưu qua TimeSeriesSplit(n=3)

---

## 4. Lý do Loại bỏ Random Forest

| Vấn đề | Chi tiết |
|:---|:---|
| $P \gg N$ | ~64 quý nhưng hàng chục biến → RF học thuộc lòng Training Set |
| OOS R² âm | Xác nhận Overfitting nặng trong thực nghiệm |
| Không giải thích được | RF Feature Importance bị nhiễu bởi đa cộng tuyến hoàn hảo |

**Giải pháp:** ElasticNet (ép coef rác về 0) + PLSR (nén dimension) → giải quyết cả $P \gg N$ lẫn đa cộng tuyến.

---
*(End of Technical Specification v2.0 — cập nhật 2026-03-18)*
