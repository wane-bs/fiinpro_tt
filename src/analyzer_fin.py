import pandas as pd
import numpy as np

def analyze_financials(fin_data_dict, output_dir):
    """
    Luồng 2: Phân tích Dữ liệu Tài chính (Financial Profiling).
    Kiểm tra sức khỏe tài chính dựa trên các chỉ tiêu có sẵn.
    """
    print("\n--- LUỒNG 2: PHÂN TÍCH BÁO CÁO TÀI CHÍNH ---")
    
    # Example: Analyze 'Kết quả kinh doanh'
    kqkd = fin_data_dict.get('Kết quả kinh doanh')
    if kqkd is not None:
        # Lấy một số chỉ tiêu quan trọng nếu có
        revenue_idx = kqkd[kqkd.iloc[:, 0].str.contains('Doanh thu bán hàng', case=False, na=False)].index
        if len(revenue_idx) > 0:
            rev_row = kqkd.iloc[revenue_idx[0], 1:]
            
            # Tính phần trăm tăng trưởng doanh thu theo Quý (QoQ)
            rev_growth = rev_row.pct_change() * 100
            print(f"Tính toán tăng trưởng doanh thu Quý / Quý (QoQ) thành công.")
            
            # Lưu lại
            growth_df = pd.DataFrame({
                'Quý': rev_row.index,
                'Doanh_Thu': rev_row.values,
                'Tang_Truong_QoQ_Pct': rev_growth.values
            })
            
            out_file = f"{output_dir}/revenue_growth.csv"
            growth_df.to_csv(out_file, index=False)
            print(f"Đã lưu báo cáo tăng trưởng doanh thu tại: {out_file}")
            # Analyze Seasonality (Hypothesis Testing)
            growth_df['Quarter_Label'] = growth_df['Quý'].str.extract(r'(Q[1-4])')
            seasonality = growth_df.groupby('Quarter_Label').agg(
                Avg_Revenue=('Doanh_Thu', 'mean'),
                Avg_Growth_Pct=('Tang_Truong_QoQ_Pct', 'mean')
            ).reset_index()
            
            # Extract Q1 and Q4 specifically
            try:
                q1_rev = seasonality[seasonality['Quarter_Label'] == 'Q1']['Avg_Revenue'].values[0]
                q4_rev = seasonality[seasonality['Quarter_Label'] == 'Q4']['Avg_Revenue'].values[0]
                q2_rev = seasonality[seasonality['Quarter_Label'] == 'Q2']['Avg_Revenue'].values[0]
                q3_rev = seasonality[seasonality['Quarter_Label'] == 'Q3']['Avg_Revenue'].values[0]
                
                if q4_rev == max(q1_rev, q2_rev, q3_rev, q4_rev) and q1_rev == min(q1_rev, q2_rev, q3_rev, q4_rev):
                    insight = f"**Chấp nhận Giả thuyết:** Tính mùa vụ thể hiện đúng. Doanh thu Q4 trung bình đạt đỉnh cao nhất ({q4_rev:,.0f} Tỷ VNĐ) và chạm mức đáy vào Q1 ({q1_rev:,.0f} Tỷ VNĐ)."
                else:
                    max_q = seasonality.loc[seasonality['Avg_Revenue'].idxmax(), 'Quarter_Label']
                    min_q = seasonality.loc[seasonality['Avg_Revenue'].idxmin(), 'Quarter_Label']
                    insight = f"**Bác bỏ Giả thuyết:** Kết quả không hoàn toàn khớp. Doanh thu đạt đỉnh vào **{max_q}** và chạm đáy vào **{min_q}**."
            except Exception as e:
                insight = "Không đủ dữ liệu của cả 4 Quý để kết luận giả thuyết."
                
            # Save insight
            with open(f"{output_dir}/revenue_seasonality_insight.txt", "w", encoding='utf-8') as f:
                f.write(insight)
                
            out_seasonality = f"{output_dir}/revenue_seasonality.csv"
            seasonality.to_csv(out_seasonality, index=False)
            print(f"Đã lưu kết quả kiểm định mùa vụ doanh thu tại: {out_seasonality}")
            
    print("Hoàn thành sơ bộ phân tích tài chính (Luồng 2).")
    return True
