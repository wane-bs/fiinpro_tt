import pandas as pd
import numpy as np
import os
from sklearn.linear_model import ElasticNetCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

class StructuralDualAuditor:
    """
    Hệ thống kiểm định chéo cấu trúc tài chính sử dụng ElasticNet và PLSR.
    Bối cảnh: P >> N, xử lý đa cộng tuyến hoàn hảo trong dữ liệu BCTC.
    """
    def __init__(self, output_dir, n_splits=5, pls_components=3):
        self.output_dir = output_dir
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
        self.pls_components = pls_components
        
        self.elastic_net = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1],
            cv=TimeSeriesSplit(n_splits=3), 
            max_iter=10000, n_jobs=-1
        )
        self.plsr = PLSRegression(n_components=self.pls_components)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        # Storage for reporting
        self.validation = []
        self.comments = []
        self.metrics_summary = []
        self.impact_results = []
        self.timeseries_data = []

    def _calculate_vip(self, pls_model):
        """Tính toán Variable Importance in Projection (VIP) cho mô hình PLSR."""
        t = pls_model.x_scores_
        w = pls_model.x_weights_
        q = pls_model.y_loadings_
        p, h = w.shape
        vips = np.zeros((p,))
        s = np.diag(t.T @ t @ q.T @ q).reshape(-1)
        total_s = np.sum(s)
        for i in range(p):
            weight = np.array([ (w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h) ])
            vips[i] = np.sqrt(p * (s.T @ weight) / total_s)
        return vips

    def _extract_target_and_features(self, df_features, df_target, target_name, exclude_keywords=None, is_bctc=True):
        """
        Trích xuất features (chuyển sang tỷ trọng %) và target.
        """
        if df_features is None or df_features.empty or df_target is None or df_target.empty:
            return None, None

        col_name_feat = df_features.columns[0]
        
        # 1. Process Features
        features_df = df_features.copy()
        
        # Loại bỏ các dòng không cần thiết nếu có exclude_keywords
        if exclude_keywords:
            exclude_idx = features_df[features_df[col_name_feat].str.contains(exclude_keywords, case=False, na=False, regex=True)].index
            if len(exclude_idx) > 0:
                features_df = features_df.drop(index=exclude_idx)
                
        features_df.set_index(col_name_feat, inplace=True)
        X_raw = features_df.T.astype(float)
        
        # Xử lý Missing values trong X
        X_raw = X_raw.dropna(axis=1, thresh=int(len(X_raw) * 0.7))
        X_raw = X_raw.fillna(0)

        # Chuyển đổi sang Common-size (Tỷ trọng %)
        # Thay vì chia cho tổng (vốn dĩ đã bị lọc bớt), ta chia cho sum() của chính các dòng được chọn để đảm bảo sum = 1
        X_pct = X_raw.div(X_raw.sum(axis=1).replace(0, 1), axis=0)
        X_pct = X_pct.fillna(0)

        # 2. Process Target
        # Định vị dòng Target
        col_name_tar = df_target.columns[0]
        target_bool = df_target[col_name_tar].str.contains(target_name, case=False, na=False, regex=True)
        if not target_bool.any():
            return None, None
            
        target_pos = target_bool.to_numpy().nonzero()[0][0]
        target_series = df_target.iloc[target_pos, 1:].astype(float).fillna(0)

        # 3. Align Index
        common_idx = X_pct.index.intersection(target_series.index)
        if len(common_idx) == 0:
            return None, None

        X = X_pct.loc[common_idx]
        y = target_series.loc[common_idx]

        return X, y

    def run_data_validation(self, data_dict):
        """Validates the input data."""
        total_missing = 0
        total_cells = 0
        status_list = []
        
        for name, df in data_dict.items():
            if df is not None and not df.empty:
                missing = df.isnull().sum().sum()
                cells = df.size
                total_missing += missing
                total_cells += cells
                missing_pct = round(missing / cells * 100, 2)
                
                status_list.append({
                    "Bảng Dữ liệu": name,
                    "Số ô trống (Missing)": missing,
                    "Tỷ lệ Missing (%)": missing_pct,
                    "Số Dòng": len(df),
                    "Số Cột": len(df.columns),
                    "Trạng thái": "Tốt" if missing_pct < 10 else "Cần làm sạch"
                })
                
        val_df = pd.DataFrame(status_list)
        val_df.to_csv(os.path.join(self.output_dir, "data_validation.csv"), index=False)
        self.validation = status_list
        
        overall_pct = round(total_missing / total_cells * 100, 2) if total_cells else 0
        self.comments.append(f"**Chất lượng dữ liệu:** Tổng quan tỷ lệ dữ liệu thiếu (missing values) là {overall_pct}%. Các bảng dữ liệu đã được tự động xử lý. Mô hình Dual-Auditor (ElasticNet & PLSR) sẵn sàng hoạt động với các biến chuẩn hóa tỷ trọng (Common-size).")

    def execute_audit(self, target_group_name, X: pd.DataFrame, y: pd.Series):
        """
        Thực thi kiểm định cuộn (Rolling Window Audit) và so sánh hiệu suất.
        """
        if X is None or X.empty or len(X) < 10:
            return

        metrics = {'EN_IS_R2': [], 'EN_OOS_R2': [], 'EN_RMSE': [],
                   'PLS_IS_R2': [], 'PLS_OOS_R2': [], 'PLS_RMSE': []}
        
        features = X.columns
        en_coefs, pls_vips = [], []

        for train_idx, test_idx in self.tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Standardize Data
            X_train_s = self.scaler_X.fit_transform(X_train)
            X_test_s = self.scaler_X.transform(X_test)
            y_train_s = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
            y_test_s = self.scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

            # --- 1. Elastic Net ---
            self.elastic_net.fit(X_train_s, y_train_s)
            y_pred_en_is = self.elastic_net.predict(X_train_s)
            y_pred_en_oos = self.elastic_net.predict(X_test_s)
            
            metrics['EN_IS_R2'].append(r2_score(y_train_s, y_pred_en_is))
            metrics['EN_OOS_R2'].append(r2_score(y_test_s, y_pred_en_oos))
            metrics['EN_RMSE'].append(root_mean_squared_error(y_test_s, y_pred_en_oos))
            en_coefs.append(self.elastic_net.coef_)

            # --- 2. PLSR ---
            # Xử lý trường hợp số lượng mẫu train nhỏ hơn số components yêu cầu
            current_n_components = min(self.pls_components, len(X_train) - 1, X_train.shape[1])
            if current_n_components < 1:
                current_n_components = 1
            
            plsr_model = PLSRegression(n_components=current_n_components)
            plsr_model.fit(X_train_s, y_train_s)
            y_pred_pls_is = plsr_model.predict(X_train_s).ravel()
            y_pred_pls_oos = plsr_model.predict(X_test_s).ravel()
            
            metrics['PLS_IS_R2'].append(r2_score(y_train_s, y_pred_pls_is))
            metrics['PLS_OOS_R2'].append(r2_score(y_test_s, y_pred_pls_oos))
            metrics['PLS_RMSE'].append(root_mean_squared_error(y_test_s, y_pred_pls_oos))
            pls_vips.append(self._calculate_vip(plsr_model))

        # --- Aggregate Metrics ---
        mean_metrics = pd.DataFrame(metrics).mean()
        self.metrics_summary.append({
            "Chỉ tiêu (Target)": target_group_name,
            "EN_IS_R2": round(mean_metrics['EN_IS_R2'], 4),
            "EN_OOS_R2": round(mean_metrics['EN_OOS_R2'], 4),
            "PLS_IS_R2": round(mean_metrics['PLS_IS_R2'], 4),
            "PLS_OOS_R2": round(mean_metrics['PLS_OOS_R2'], 4),
            "Số mẫu (Quarters)": len(X)
        })

        # --- Aggregate Impact ---
        mean_en_coefs = np.mean(en_coefs, axis=0)
        mean_pls_vips = np.mean(pls_vips, axis=0)
        
        impact_df = pd.DataFrame({
            'Target': target_group_name,
            'Feature': features,
            'ElasticNet_Mean_Coef': mean_en_coefs,
            'PLSR_Mean_VIP': mean_pls_vips
        }).sort_values('PLSR_Mean_VIP', ascending=False)
        
        self.impact_results.append(impact_df.head(10)) # Keep top 10

        # --- Phân tích Kết quả (Lớp 1 & Lớp 2) ---
        self._generate_audit_insights(target_group_name, mean_metrics, impact_df)
        self._prepare_timeseries_data(target_group_name, X, y, impact_df)

    def _generate_audit_insights(self, target_group_name, mean_metrics, impact_df):
        en_oos = mean_metrics['EN_OOS_R2']
        pls_oos = mean_metrics['PLS_OOS_R2']
        
        insight = f"**Báo cáo Đối chiếu mục tiêu `{target_group_name}`:**\n"
        
        # Lớp 1: Khả năng tổng quát hóa
        insight += f"- **Hiệu suất (OOS R2):** ElasticNet đạt {en_oos:.4f}, PLSR đạt {pls_oos:.4f}. "
        if en_oos > 0.3 and pls_oos > 0.3:
            insight += "✅ Trọng số tác động ổn định cao (OOS R2 > 0.3). Cấu trúc nhân tố rõ ràng và nhất quán qua thời gian.\n"
        elif en_oos > 0 and pls_oos > 0:
            insight += "⚠️ Trọng số tác động tương đối ổn định (OOS R2 dương). Nhân tố cơ cấu có tác động nhưng biến động giữa các giai đoạn.\n"
        else:
            insight += "❌ Trọng số tác động không ổn định (OOS R2 âm). Cấu trúc nhân tố thay đổi lớn giữa các giai đoạn — cần diễn giải thận trọng.\n"

        if en_oos > pls_oos + 0.1:
            insight += "  - *Kết luận Lớp 1:* ElasticNet áp đảo. BCTC chứa nhiều khoản mục rác/ít thay đổi, mô hình đã thành công ép hệ số rác về 0.\n"
        elif pls_oos > en_oos + 0.1:
            insight += "  - *Kết luận Lớp 1:* PLSR áp đảo. Hiện tượng đồng pha/đa cộng tuyến (Group Effect) giữa các khoản mục rất mạnh.\n"
        else:
            insight += "  - *Kết luận Lớp 1:* Hai mô hình ngang tài. Tín hiệu khá rõ ràng và ít bị nhiễu do cộng tuyến cục bộ.\n"

        # Lớp 2: Yếu tố cấu trúc
        consensus = impact_df[(impact_df['PLSR_Mean_VIP'] > 1.0) & (impact_df['ElasticNet_Mean_Coef'].abs() > 0.05)]
        divergence = impact_df[(impact_df['PLSR_Mean_VIP'] > 1.0) & (impact_df['ElasticNet_Mean_Coef'].abs() <= 0.05)]
        
        insight += "- **Tín hiệu Cấu trúc Cốt lõi (Lớp 2):**\n"
        if not consensus.empty:
            top_con = consensus.iloc[0]
            insight += f"  - 🌟 **Đồng thuận (Drivers chính):** '{top_con['Feature']}' là nhân tố quyết định lớn nhất (VIP={top_con['PLSR_Mean_VIP']:.2f}, Coef={top_con['ElasticNet_Mean_Coef']:.2f}).\n"
        
        if not divergence.empty:
            top_div = divergence.iloc[0]
            insight += f"  - 🔗 **Phân kỳ (Đóng góp Nhóm):** '{top_div['Feature']}' bị ElasticNet bỏ qua (Coef xấp xỉ 0) nhưng vô cùng quan trọng đối với PLSR (VIP={top_div['PLSR_Mean_VIP']:.2f}). Điều này cho thấy khoản mục này luôn biến động cùng chiều với một chỉ tiêu lớn khác trong cấu trúc.\n"
            
        self.comments.append(insight)

    def _prepare_timeseries_data(self, target_group_name, X_pct, y, impact_df):
        top5_features = impact_df.head(5)['Feature'].tolist()
        
        ts_data = []
        for q in X_pct.index:
            # Target
            ts_data.append({
                "Target_Name": target_group_name,
                "Quarter": q,
                "Variable_Type": "Target_Performance",
                "Variable_Name": target_group_name,
                "Value": y.loc[q]
            })
            # Top 5 Features (Tỷ trọng %)
            for feat in top5_features:
                ts_data.append({
                    "Target_Name": target_group_name,
                    "Quarter": q,
                    "Variable_Type": "Feature_Ratio",
                    "Variable_Name": feat,
                    "Value": X_pct.loc[q, feat] * 100 # Chuyển thành tỷ lệ %
                })
        self.timeseries_data.extend(ts_data)

    def process_all_targets(self, data_dict):
        """Pipelines validation, modeling, and exporting."""
        print("Đang kiểm định dữ liệu...")
        self.run_data_validation(data_dict)
        
        bctc_df = data_dict.get('Bảng cân đối kế toán')
        kqkd_df = data_dict.get('Kết quả kinh doanh')
        
        if bctc_df is not None and kqkd_df is not None:
            col_bctc = bctc_df.columns[0]
            col_kqkd = kqkd_df.columns[0]
            
            # Helper to find rows
            def _get_row(df, col, pattern):
                mask = df[col].str.contains(pattern, case=False, na=False, regex=True)
                if mask.any():
                    idx = mask.to_numpy().nonzero()[0][0]
                    return df.iloc[idx, 1:].astype(float).fillna(0)
                return None

            # Tính toán các chỉ số cơ bản
            net_income = _get_row(kqkd_df, col_kqkd, r'Lãi/\(lỗ\) thuần sau thuế|Lợi nhuận sau thuế|Lợi nhuận thuần')
            total_assets = _get_row(bctc_df, col_bctc, 'TỔNG CỘNG TÀI SẢN')
            total_equity = _get_row(bctc_df, col_bctc, 'VỐN CHỦ SỞ HỮU')
            gross_profit = _get_row(kqkd_df, col_kqkd, r'Lợi nhuận gộp')
            net_revenue = _get_row(kqkd_df, col_kqkd, r'Doanh thu thuần|Doanh thu bán hàng')

            target_bool = bctc_df[col_bctc].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False, regex=True)
            if target_bool.any():
                split_pos = target_bool.to_numpy().nonzero()[0][0]
                assets_df = bctc_df.iloc[:split_pos+1].copy()
                capital_df = bctc_df.iloc[split_pos+1:].copy()

                # --- 1. MẢNG TÀI SẢN (Target: ROA) ---
                print("Đang chạy Dual-Auditor cho Mảng Tài sản...")
                if net_income is not None and total_assets is not None:
                    roa_series = (net_income / total_assets.replace(0, np.nan)) * 100
                    roa_series = roa_series.fillna(0)
                    # Tạo dataframe giả để khớp API _extract_target_and_features
                    target_df_roa = pd.DataFrame({'Tên': ['Chỉ số ROA Tùy chỉnh']})
                    target_df_roa = pd.concat([target_df_roa, roa_series.to_frame().T.reset_index(drop=True)], axis=1)

                    X_a, y_a = self._extract_target_and_features(
                        df_features=assets_df, 
                        df_target=target_df_roa, 
                        target_name='Chỉ số ROA Tùy chỉnh',
                        exclude_keywords='TÀI SẢN NGẮN HẠN|TÀI SẢN DÀI HẠN|TỔNG CỘNG TÀI SẢN'
                    )
                    self.execute_audit('Mảng Tài Sản -> Hiệu quả ROA', X_a, y_a)

                # --- 2. MẢNG NGUỒN VỐN (Target: ROE) ---
                print("Đang chạy Dual-Auditor cho Mảng Nguồn vốn...")
                if net_income is not None and total_equity is not None:
                    roe_series = (net_income / total_equity.replace(0, np.nan)) * 100
                    roe_series = roe_series.fillna(0)
                    target_df_roe = pd.DataFrame({'Tên': ['Chỉ số ROE Tùy chỉnh']})
                    target_df_roe = pd.concat([target_df_roe, roe_series.to_frame().T.reset_index(drop=True)], axis=1)

                    X_c, y_c = self._extract_target_and_features(
                        df_features=capital_df,
                        df_target=target_df_roe,
                        target_name='Chỉ số ROE Tùy chỉnh',
                        exclude_keywords='NỢ PHẢI TRẢ|VỐN CHỦ SỞ HỮU|TỔNG CỘNG NGUỒN VỐN|Lợi nhuận sau thuế|LNST|chưa phân phối'
                    )
                    if X_c is not None and len(X_c) > 0:
                         self.execute_audit('Mảng Nguồn Vốn -> Hiệu quả ROE', X_c, y_c)
        
        # --- 3. MẢNG DOANH THU CẤU THÀNH (Target: Biên LN Gộp) ---
        print("Đang chạy Dual-Auditor cho Cấu trúc Doanh thu...")
        if kqkd_df is not None and gross_profit is not None and net_revenue is not None:
            margin_series = (gross_profit / net_revenue.replace(0, np.nan)) * 100
            margin_series = margin_series.fillna(0)
            target_df_margin = pd.DataFrame({'Tên': ['Biên LN Gộp']})
            target_df_margin = pd.concat([target_df_margin, margin_series.to_frame().T.reset_index(drop=True)], axis=1)

            X_rev, y_rev = self._extract_target_and_features(
                df_features=kqkd_df,
                df_target=target_df_margin,
                target_name='Biên LN Gộp',
                exclude_keywords=r'Lợi nhuận gộp|Tổng lợi nhuận kế toán|Lợi nhuận thuần|Lãi/\(lỗ\)'
            )
            self.execute_audit('Cấu trúc KQKD -> Biên LN Gộp (%)', X_rev, y_rev)
        
        self.export_results()
        self.generate_markdown_report()
        
    def export_results(self):
        if self.impact_results:
            final_impact_df = pd.concat(self.impact_results, ignore_index=True)
            final_impact_df.to_csv(os.path.join(self.output_dir, "structure_impact.csv"), index=False)
            
        if self.metrics_summary:
            metrics_df = pd.DataFrame(self.metrics_summary)
            metrics_df.to_csv(os.path.join(self.output_dir, "model_metrics.csv"), index=False)
            
        if self.timeseries_data:
            ts_df = pd.DataFrame(self.timeseries_data)
            ts_df.to_csv(os.path.join(self.output_dir, "structure_timeseries.csv"), index=False)
            
    def generate_markdown_report(self):
        def df_to_markdown(df):
            if df.empty: return ""
            headers = "| " + " | ".join(df.columns) + " |"
            sep = "|-" + "-|-".join(["-"] * len(df.columns)) + "-|"
            rows = []
            for _, row in df.iterrows():
                # Format numbers
                formatted_values = []
                for val in row.values:
                    if isinstance(val, (float, np.float64)):
                        formatted_values.append(f"{val:.4f}")
                    else:
                        formatted_values.append(str(val))
                rows.append("| " + " | ".join(formatted_values) + " |")
            return "\n".join([headers, sep] + rows)

        report_path = os.path.join(self.output_dir, "analysis_report.md")
        with open(report_path, "w", encoding='utf-8') as f:
            f.write("# Báo cáo Phân tích Cơ cấu Tài chính (Dual-Auditor)\n\n")
            f.write("> Báo cáo này phân tích **tác động cơ cấu** (Impact Weights) thay vì dự báo. Kiến trúc Đối chiếu Kép (ElasticNet + PLSR) xác định các nhân tố cấu trúc trọng yếu và triệt tiêu hiệu ứng Đa cộng tuyến hoàn hảo.\n\n")
            
            f.write("## 1. Kiểm định chất lượng dữ liệu\n")
            if self.comments:
                f.write(self.comments[0] + "\n\n")
            
            f.write("## 2. Độ tin cậy của Trọng số Tác động (OOS R2)\n")
            f.write("OOS R2 là thước đo **độ ổn định** của Impact Weights qua thời gian, không phải sức mạnh dự báo. R2 dương cho thấy cấu trúc nhân tố ổn định.\n\n")
            if self.metrics_summary:
                metrics_df = pd.DataFrame(self.metrics_summary)
                f.write(df_to_markdown(metrics_df) + "\n\n")
            
            f.write("### Phân tích Chuyên sâu:\n")
            for c in self.comments[1:]:
                f.write(c + "\n")
            
            f.write("\n## 3. Các Yếu tố Cấu trúc Trọng yếu (Top Drivers)\n")
            if self.impact_results:
                final_impact_df = pd.concat(self.impact_results, ignore_index=True)
                for target in final_impact_df['Target'].unique():
                    f.write(f"### Tác động tới `{target}`\n")
                    f.write("*Yêu cầu: PLS VIP > 1.0 biểu thị mức độ quan trọng cao. ElasticNet Coef xác định chiều hướng (+/-) tác động.*\n\n")
                    subset = final_impact_df[final_impact_df['Target'] == target]
                    f.write(df_to_markdown(subset[['Feature', 'PLSR_Mean_VIP', 'ElasticNet_Mean_Coef']]))
                    f.write("\n\n")
                    
            f.write("## 4. Kiểm định giả thuyết Mùa vụ\n")
            price_insight_path = os.path.join(self.output_dir, "price_seasonality_insight.txt")
            if os.path.exists(price_insight_path):
                with open(price_insight_path, "r", encoding="utf-8") as f_in:
                    f.write("### Biến động Giá & Động lượng\n")
                    f.write("- " + f_in.read().strip() + "\n\n")
                    
            rev_insight_path = os.path.join(self.output_dir, "revenue_seasonality_insight.txt")
            if os.path.exists(rev_insight_path):
                with open(rev_insight_path, "r", encoding="utf-8") as f_in:
                    f.write("### Doanh thu & Sức khỏe Tài chính\n")
                    f.write("- " + f_in.read().strip() + "\n\n")
                    
        print(f"Đã tạo báo cáo Dual-Auditor tại: {report_path}")

def run_structure_analysis(data_dict, output_dir):
    print("\n--- LUỒNG 4: PHÂN TÍCH CẤU TRÚC VÀ TÁC ĐỘNG (DUAL-AUDITOR) ---")
    analyzer = StructuralDualAuditor(output_dir)
    analyzer.process_all_targets(data_dict)
    print("Hoàn thành phân tích cấu trúc kép (Luồng 4).")
    return True
