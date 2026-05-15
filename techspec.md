# Technical Specification — Fiinpro Analysis Pipeline (v3.0)

Tài liệu diễn giải kiến trúc, logic tính toán và Data Flow của dự án sau khi tái cấu trúc sang triết lý **Phân tích Cơ cấu (Structural Analysis)** và bổ sung các module định giá nâng cao.

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
    ┌───────────────────────────────────────────────┐
    │  Chương 1  │  Chương 2  │  Chương 3  │ Chương 4│
    │  Nền tảng  │  Hiệu suất │  Định giá  │ Cấu trúc│
    │  Tĩnh      │  + Chu kỳ  │  + Độ nhạy │ + Chu kỳ│
    └───────────────────────────────────────────────┘
                         ↓
                  signal_engine.py
                         ↓
                 Composite Score v2 (3 Pillars)
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
- Quét sheet "Thuyết minh" bằng Regex tìm keyword chính sách kế toán
- Output: `preanalysis_report.json`

**`analyzer_ratios.py` — Balanced Vertical Analysis (Module 1.3) [UPDATE v3.0]**
Hệ thống sử dụng cơ chế **Biến bù trừ (Balancing Items)** để đảm bảo tổng cơ cấu luôn chính xác 100.0%:
- **Tài sản:** 
  - `TSNH khác` = `TSNH` - (`Tiền` + `Phải thu` + `HTK`)
  - `TSDH khác` = `TSDH` - `TSCĐ`
- **Nguồn vốn:**
  - `Nguồn vốn khác` = `Tổng nguồn vốn` - (`Nợ ngắn hạn` + `Nợ dài hạn` + `Vốn CSH`)
- **KQKD (Allocation of Revenue):**
  - Chuyển `Giá vốn`, `CP Bán hàng`, `CP Quản lý` sang trị tuyệt đối (dương).
  - `Thuế & CP Khác (Ròng)` = `Doanh thu thuần` - (`Σ Chi phí` + `Lợi nhuận ròng`).
- Output: `vertical_analysis.csv`

**`analyzer_ratios.py` — Solvency & Liquidity (Module 1.4)**
```
Current Ratio     = TSNH / NNH
Quick Ratio       = (TSNH - HTK) / NNH
D/E Ratio         = Tổng Nợ / Vốn CSH
Debt Ratio        = Tổng Nợ / Tổng TS
Interest Coverage = EBIT / |Lãi vay|
```

---

### Bước 2 — Chương 2: Hiệu suất Động

**`analyzer_dupont.py` — DuPont 3 Nhân tố (Module 2.3)**
```
ROE = Net Margin × Asset Turnover × Equity Multiplier
    = (NI/Rev)  × (Rev/AvgAssets) × (AvgAssets/AvgEquity)
```
- Delta ROE = ROE_t − ROE_{t-1} → Waterfall chart

**`analyzer_cashflow.py` — Cash Flow Quality (Module 2.4)**
```
CFO/NI Ratio  = CFO / Net Income          (ngưỡng cảnh báo: < 0.7 liên tiếp 3Q)
FCF           = CFO − |Capex|
Accrual Ratio = (NI − CFO) / Avg Assets   (ngưỡng cảnh báo: > 5%)
```

---

### Bước 3 — Chương 3: Định giá & Độ nhạy [UPDATE v3.0]

**`analyzer_dcf.py` — DCF Analysis**
- Bước 1: Tính FCFF lịch sử từ EBIT, Tax (20%), Khấu hao và ΔWC.
- Bước 2: Tính WACC từ CAPM (r_f=4.5%, ERP=7%) và chi phí lãi vay thực tế.
- Bước 3: PV 5 năm + Terminal Value (g_tv=2-3%).

**`analyzer_sotp.py` — Sum-of-the-parts Valuation [NEW]**
Định giá dựa trên dữ liệu phân đoạn kinh doanh:
- `Công nghệ`: P/E Multiples.
- `Viễn thông`: EV/EBITDA Multiples.
- `Giáo dục & Khác`: P/S Multiples.
- `Equity Value` = Σ(EV Segments) - Net Debt.

**P/E Forward Band Analysis [NEW]**
- `Historical Band`: Mean P/E ± 2 Standard Deviation (Dải 5 năm).
- `Forward EPS`: Dự phóng dựa trên CAGR 3 năm gần nhất.

---

### Bước 4 — Chương 4: Cấu trúc & Chu kỳ

**`analyzer_cycle.py` — STL + CCF (Module 4a)**
- STL Decomposition: Tách Trend, Seasonal (4Q), Residual.
- Cross-Correlation (CCF): Tìm độ trễ (Lag -4 đến +4) giữa các chỉ số vĩ mô/ngành và nội tại.

**`analyzer_structure.py` — Dual-Auditor (Module 4b)**
Sử dụng ElasticNet (Regularization) và PLSR (Dimension Reduction) để xử lý bài toán $P \gg N$:
- **ElasticNet:** Xác định chiều hướng (+/-) và loại bỏ biến nhiễu.
- **PLSR:** Tính toán VIP Score (Tầm quan trọng biến trong không gian Latent).
- **Quy tắc:** `VIP > 1.0` + `|Coef| >> 0` → Key Structural Driver.

---

### Bước 5 — Tổng hợp Tín hiệu (Composite Score v2) [UPDATE v3.0]

Chuyển đổi sang hệ thống **3 Trụ cột** với trọng số:
1. **Sức khỏe (30%):** Thanh khoản, Đòn bẩy, Chất lượng dòng tiền.
2. **Tăng trưởng (35%):** Tốc độ DT/LN, Structural Drivers từ Chương 4.
3. **Định giá (35%):** DCF, SoTP, P/E Band, Valuation Bands.

---

### Bước 6 — Dashboard Streamlit

| Tab | Nội dung cập nhật |
|:---|:---|
| 📘 Chương 1 | Audit Gauge (No-cache) + Balanced Common-size (100%) |
| 📗 Chương 2 | DuPont Waterfall + Accrual + Cycle Overlay |
| 📙 Chương 3 | SOTP Waterfall + P/E Forward Band + DCF Heatmap |
| 📕 Chương 4 | STL Decomposition + CCF Lead-Lag + VIP Analysis |
| 📋 Báo cáo | Composite Score v2 Gauge + Summary Recommendation |

---

## 3. Xử lý Dữ liệu & Lý do kỹ thuật
- **DivideByZero:** Luôn sử dụng `np.where` để tránh lỗi mẫu số bằng 0.
- **Random Forest:** Đã bị loại bỏ hoàn toàn do hiện tượng Overfitting mạnh trên tập dữ liệu chuỗi thời gian ngắn (N ≈ 64).
- **STL Fallback:** Sử dụng Rolling Mean nếu dữ liệu ít hơn 8 quý.

---
*(End of Technical Specification v3.0 — cập nhật 2026-05-15)*
