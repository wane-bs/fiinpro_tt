# Báo cáo Phân tích Cơ cấu Tài chính (Dual-Auditor)

> Báo cáo này phân tích **tác động cơ cấu** (Impact Weights) thay vì dự báo. Kiến trúc Đối chiếu Kép (ElasticNet + PLSR) xác định các nhân tố cấu trúc trọng yếu và triệt tiêu hiệu ứng Đa cộng tuyến hoàn hảo.

## 1. Kiểm định chất lượng dữ liệu
**Chất lượng dữ liệu:** Tổng quan tỷ lệ dữ liệu thiếu (missing values) là 0.0%. Các bảng dữ liệu đã được tự động xử lý. Mô hình Dual-Auditor (ElasticNet & PLSR) sẵn sàng hoạt động với các biến chuẩn hóa tỷ trọng (Common-size).

## 2. Độ tin cậy của Trọng số Tác động (OOS R2)
OOS R2 là thước đo **độ ổn định** của Impact Weights qua thời gian, không phải sức mạnh dự báo. R2 dương cho thấy cấu trúc nhân tố ổn định.

| Chỉ tiêu (Target) | EN_IS_R2 | EN_OOS_R2 | PLS_IS_R2 | PLS_OOS_R2 | Số mẫu (Quarters) |
|---|---|---|---|---|---|
| Mảng Tài Sản -> Hiệu quả ROA | 0.4838 | -8.4466 | 0.6356 | -79.7117 | 64 |
| Mảng Nguồn Vốn -> Hiệu quả ROE | 0.7891 | -16.0310 | 0.8513 | -235.4287 | 64 |
| Cấu trúc KQKD -> Biên LN Gộp (%) | 0.9995 | 0.6006 | 0.9901 | 0.4337 | 64 |

### Phân tích Chuyên sâu:
**Báo cáo Đối chiếu mục tiêu `Mảng Tài Sản -> Hiệu quả ROA`:**
- **Hiệu suất (OOS R2):** ElasticNet đạt -8.4466, PLSR đạt -79.7117. ❌ Trọng số tác động không ổn định (OOS R2 âm). Cấu trúc nhân tố thay đổi lớn giữa các giai đoạn — cần diễn giải thận trọng.
  - *Kết luận Lớp 1:* ElasticNet áp đảo. BCTC chứa nhiều khoản mục rác/ít thay đổi, mô hình đã thành công ép hệ số rác về 0.
- **Tín hiệu Cấu trúc Cốt lõi (Lớp 2):**
  - 🌟 **Đồng thuận (Drivers chính):** '     Dự phòng giảm giá HTK' là nhân tố quyết định lớn nhất (VIP=1.88, Coef=0.05).
  - 🔗 **Phân kỳ (Đóng góp Nhóm):** 'Các khoản phải thu' bị ElasticNet bỏ qua (Coef xấp xỉ 0) nhưng vô cùng quan trọng đối với PLSR (VIP=1.32). Điều này cho thấy khoản mục này luôn biến động cùng chiều với một chỉ tiêu lớn khác trong cấu trúc.

**Báo cáo Đối chiếu mục tiêu `Mảng Nguồn Vốn -> Hiệu quả ROE`:**
- **Hiệu suất (OOS R2):** ElasticNet đạt -16.0310, PLSR đạt -235.4287. ❌ Trọng số tác động không ổn định (OOS R2 âm). Cấu trúc nhân tố thay đổi lớn giữa các giai đoạn — cần diễn giải thận trọng.
  - *Kết luận Lớp 1:* ElasticNet áp đảo. BCTC chứa nhiều khoản mục rác/ít thay đổi, mô hình đã thành công ép hệ số rác về 0.
- **Tín hiệu Cấu trúc Cốt lõi (Lớp 2):**
  - 🌟 **Đồng thuận (Drivers chính):** '     Thuế và các khoản phải trả Nhà nước' là nhân tố quyết định lớn nhất (VIP=2.02, Coef=0.46).
  - 🔗 **Phân kỳ (Đóng góp Nhóm):** 'Nợ dài hạn' bị ElasticNet bỏ qua (Coef xấp xỉ 0) nhưng vô cùng quan trọng đối với PLSR (VIP=1.72). Điều này cho thấy khoản mục này luôn biến động cùng chiều với một chỉ tiêu lớn khác trong cấu trúc.

**Báo cáo Đối chiếu mục tiêu `Cấu trúc KQKD -> Biên LN Gộp (%)`:**
- **Hiệu suất (OOS R2):** ElasticNet đạt 0.6006, PLSR đạt 0.4337. ✅ Trọng số tác động ổn định cao (OOS R2 > 0.3). Cấu trúc nhân tố rõ ràng và nhất quán qua thời gian.
  - *Kết luận Lớp 1:* ElasticNet áp đảo. BCTC chứa nhiều khoản mục rác/ít thay đổi, mô hình đã thành công ép hệ số rác về 0.
- **Tín hiệu Cấu trúc Cốt lõi (Lớp 2):**
  - 🌟 **Đồng thuận (Drivers chính):** 'Chi phí quản lý doanh  nghiệp' là nhân tố quyết định lớn nhất (VIP=1.50, Coef=-0.23).
  - 🔗 **Phân kỳ (Đóng góp Nhóm):** '     Thuế thu nhập doanh nghiệp – hiện thời' bị ElasticNet bỏ qua (Coef xấp xỉ 0) nhưng vô cùng quan trọng đối với PLSR (VIP=1.18). Điều này cho thấy khoản mục này luôn biến động cùng chiều với một chỉ tiêu lớn khác trong cấu trúc.


## 3. Các Yếu tố Cấu trúc Trọng yếu (Top Drivers)
### Tác động tới `Mảng Tài Sản -> Hiệu quả ROA`
*Yêu cầu: PLS VIP > 1.0 biểu thị mức độ quan trọng cao. ElasticNet Coef xác định chiều hướng (+/-) tác động.*

| Feature | PLSR_Mean_VIP | ElasticNet_Mean_Coef |
|---|---|---|
|      Dự phòng giảm giá HTK | 1.8773 | 0.0514 |
|      Đầu tư vào các công ty liên kết | 1.7083 | 0.0701 |
| Đầu tư dài hạn | 1.6953 | 0.0557 |
|      Phải thu khách hàng | 1.6929 | 0.1299 |
|      GTCL tài sản cố định vô hình | 1.6857 | 0.0858 |
|      Chi phí trả trước ngắn hạn | 1.3679 | -0.1155 |
| Các khoản phải thu | 1.3196 | 0.0154 |
|           Nguyên giá TSCĐ vô hình | 1.2952 | 0.0567 |
|      Trả trước dài hạn | 1.2903 | -0.0059 |
|      Phải thu thuế khác | 1.2773 | 0.0485 |

### Tác động tới `Mảng Nguồn Vốn -> Hiệu quả ROE`
*Yêu cầu: PLS VIP > 1.0 biểu thị mức độ quan trọng cao. ElasticNet Coef xác định chiều hướng (+/-) tác động.*

| Feature | PLSR_Mean_VIP | ElasticNet_Mean_Coef |
|---|---|---|
|      Thuế và các khoản phải trả Nhà nước | 2.0204 | 0.4572 |
|      Thặng dư vốn cổ phần | 1.7776 | 0.0715 |
| Nợ dài hạn | 1.7245 | 0.0347 |
|      Vay dài hạn | 1.6636 | 0.0128 |
|      Quỹ khác | 1.4506 | 0.0572 |
| Vốn Ngân sách nhà nước và quỹ khác | 1.4072 | -0.0015 |
|      Vốn ngân sách nhà nước | 1.4072 | -0.0015 |
| Vốn và các quỹ | 1.3464 | -0.0130 |
|      Quỹ đầu tư và phát triển | 1.3459 | -0.0043 |
|      Cổ phiếu quỹ | 1.3392 | -0.0232 |

### Tác động tới `Cấu trúc KQKD -> Biên LN Gộp (%)`
*Yêu cầu: PLS VIP > 1.0 biểu thị mức độ quan trọng cao. ElasticNet Coef xác định chiều hướng (+/-) tác động.*

| Feature | PLSR_Mean_VIP | ElasticNet_Mean_Coef |
|---|---|---|
| Chi phí quản lý doanh  nghiệp | 1.5031 | -0.2263 |
| Giá vốn hàng bán | 1.4829 | 0.5159 |
| Lợi nhuận của Cổ đông của Công ty mẹ | 1.4211 | 0.1262 |
|      Thuế thu nhập doanh nghiệp – hiện thời | 1.1764 | -0.0243 |
| Chi phí bán hàng | 1.1722 | -0.1785 |
| Lợi ích của cổ đông thiểu số | 1.1143 | 0.0755 |
| Doanh thu thuần | 1.0781 | 0.0000 |
| Chi phí thuế thu nhập doanh nghiệp | 1.0340 | -0.0128 |
| Doanh thu bán hàng và cung cấp dịch vụ | 1.0133 | 0.0000 |
| Các khoản giảm trừ doanh thu | 0.7846 | -0.0368 |

## 4. Kiểm định giả thuyết Mùa vụ
### Biến động Giá & Động lượng
- **Chấp nhận Giả thuyết:** Tính thời vụ tháng 4-5 thể hiện rõ ràng. Trung bình tăng trưởng tháng 4-5 (0.2173%) **cao hơn** so với trung bình các tháng còn lại trong năm (0.0503%).

### Doanh thu & Sức khỏe Tài chính
- **Chấp nhận Giả thuyết:** Tính mùa vụ thể hiện đúng. Doanh thu Q4 trung bình đạt đỉnh cao nhất (12,297 Tỷ VNĐ) và chạm mức đáy vào Q1 (9,377 Tỷ VNĐ).

