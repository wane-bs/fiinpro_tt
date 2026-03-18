import pandas as pd
import numpy as np

def run_causal_analysis(fin_data_dict, price_df, output_dir):
    """
    Luồng 3: Phân tích Tương quan chéo (Causal Cross-Analysis).
    Sử dụng Time-Window Aggregation để map dữ liệu Giá vào Quý Tài chính,
    bảo toàn định dạng gốc của cả hai.
    """
    print("\n--- LUỒNG 3: PHÂN TÍCH TƯƠNG QUAN CHÉO (CROSS-ANALYSIS) ---")
    
    # 1. Định nghĩa các Quý dựa trên Dữ liệu Giá
    # Quý 1: T1-T3, Quý 2: T4-T6, Quý 3: T7-T9, Quý 4: T10-T12
    if 'NGÀY' not in price_df.columns:
        print("Không tìm thấy cột NGÀY trong dữ liệu Giá. Bỏ qua Luồng 3.")
        return
        
    # Tạo nhãn Quý cho dữ liệu Giá giống format BCTC: "Q1/2010"
    price_df['Quarter'] = price_df['NGÀY'].dt.quarter
    price_df['Year'] = price_df['NGÀY'].dt.year
    price_df['BCTC_Quarter_Label'] = "Q" + price_df['Quarter'].astype(str) + "/" + price_df['Year'].astype(str)
    
    # 2. Time-Window Aggregation: Tính Trung bình Khối lượng và Giá theo Quý
    # Điều này tạo ra một bảng dữ liệu gộp không phá vỡ logic chuỗi thời gian phân tách
    agg_funcs = {
        'KL KHỚP': 'mean',
        'GIÁ': 'mean'
    }
    
    if 'VOLATILITY_30D' in price_df.columns:
        agg_funcs['VOLATILITY_30D'] = 'mean'
        
    quarterly_price_stats = price_df.groupby('BCTC_Quarter_Label').agg(agg_funcs).reset_index()
    
    # Rename columns for clarity
    quarterly_price_stats = quarterly_price_stats.rename(columns={
        'KL KHỚP': 'TB_KhoiLuong_KhopLenh_Ngay',
        'GIÁ': 'TB_Gia_Ngay',
        'VOLATILITY_30D': 'TB_BienDong'
    })
    
    out_file = f"{output_dir}/aggregated_price_by_quarter.csv"
    quarterly_price_stats.to_csv(out_file, index=False)
    
    print(f"Đã tạo bảng gom nhóm Dữ liệu Giá theo nhãn Quý (Time-Window Aggregation).")
    print(f"Bảng này có thể dùng chéo với các cột Quý của BCTC (Q1/2010...) để chạy kiểm định Granger Causality.")
    print(f"Đã lưu tại: {out_file}")
    
    return quarterly_price_stats
