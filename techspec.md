# Technical Specification — Fiinpro Analysis Pipeline (v3.0)

Tài liệu diễn giải kiến trúc, logic tính toán và Data Flow của dự án. Cập nhật mới nhất tập trung vào tính chính xác của **Cơ cấu 100% (Balanced Structure)** và các module định giá nâng cao.

---

## 1. Cải tiến Logic Phân tích Cơ cấu (Vertical Analysis)

Để đảm bảo các biểu đồ Stacked Bar luôn đạt mốc 100.0% và phản ánh đúng bản chất tài chính, hệ thống áp dụng cơ chế **Biến bù trừ (Balancing Items)**:

### 1.1 Mảng Tài sản & Nguồn vốn
- **Tài sản ngắn hạn khác** = `TÀI SẢN NGẮN HẠN` - (`Tiền` + `Phải thu` + `Hàng tồn kho`).
- **Tài sản dài hạn khác** = `TÀI SẢN DÀI HẠN` - `Tài sản cố định`.
- **Nguồn vốn khác** = `TỔNG CỘNG NGUỒN VỐN` - (`Nợ ngắn hạn` + `Nợ dài hạn` + `Vốn chủ sở hữu`).

### 1.2 Mảng Kết quả Kinh doanh (Allocation of Revenue)
Hệ thống chuyển đổi sang mô hình "Phân bổ Doanh thu":
- Các khoản chi phí (`Giá vốn`, `CP Bán hàng`, `CP Quản lý`) được lấy **Trị tuyệt đối** để xếp chồng dương.
- **Thuế & CP Khác (Ròng)** = `Doanh thu thuần` - (`|Giá vốn|` + `|CP BH|` + `|CP QL|` + `Lợi nhuận sau thuế`).
- **Lợi nhuận sau thuế:** Giữ nguyên dấu (dương là lãi, âm là lỗ).
- *Kết quả:* Nếu lãi, tổng đạt 100%. Nếu lỗ, khối chi phí > 100% và lợi nhuận cắm âm xuống dưới trục 0.

---

## 2. Module Định giá Bổ sung (Chương 3)

### 3.5 Sum-of-the-parts (SoTP) Valuation
Định giá từng phần dựa trên dữ liệu phân đoạn (Segments) từ IR:
- **Mảng Công nghệ:** Áp dụng P/E Target × Lợi nhuận mảng.
- **Mảng Viễn thông:** Áp dụng EV/EBITDA Target × EBITDA mảng.
- **Mảng Giáo dục & Khác:** Áp dụng P/S Multiple.
- **Equity Value** = `Σ(Enterprise Value mảng)` - `Net Debt`.

### 3.6 P/E Forward Band Analysis
Kết hợp dữ liệu lịch sử và kỳ vọng tương lai:
- **Historical Band:** Rolling Mean P/E 5 năm ± 1.5-2.0 Standard Deviation.
- **Forward EPS:** Ước tính dựa trên CAGR 3 năm gần nhất hoặc kế hoạch kinh doanh.
- **Valuation Zone:** Xác định vị thế giá hiện tại so với biên an toàn của dải định giá dự phóng.

---

## 3. Hệ thống Tín hiệu Composite Score v2

Chuyển đổi từ hệ thống điểm đều sang hệ thống **3 Trụ cột (3 Pillars)** với trọng số tùy chỉnh:

| Trụ cột | Trọng số | Thành phần chính |
|:---|:---:|:---|
| **P1: Sức khỏe (Health)** | 30% | Thanh khoản, Đòn bẩy, Chất lượng dòng tiền (CFO/NI) |
| **P2: Tăng trưởng (Growth)** | 35% | Tăng trưởng DT/LN, Momentum (ElasticNet/PLSR drivers) |
| **P3: Định giá (Valuation)** | 35% | DCF Heatmap, Valuation Bands, SoTP, P/E Band |

**Công thức:** `Composite Score = Σ (Score_Pi × Weight_Pi)` (Thang điểm -100 đến +100).

---

## 4. Cải tiến Dashboard (Streamlit)

- **Audit Reliability Gauge:** Sử dụng `@st.cache_data(ttl=1)` để đảm bảo real-time update dữ liệu từ pipeline.
- **Dark-mode optimized charts:** Toàn bộ Plotly charts sử dụng bảng màu HSL và độ tương phản cao cho chế độ tối.

---
*(End of Technical Specification v3.0 — cập nhật 2026-05-15)*
