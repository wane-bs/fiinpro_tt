import pandas as pd
import numpy as np

def preprocess_data(data):
    """
    Preprocess financial and price data according to assessment rules.
    Takes the dictionary from data_loader.
    """
    # 1. Financial Preprocessing
    # We don't drop specific rows like 'Dự phòng trợ cấp thôi việc'.
    # We do fill specific sparse NaN values in 'Thuyết minh' context.
    
    finance_sheets = ['Bảng cân đối kế toán', 'Kết quả kinh doanh', 'Lưu chuyển tiền tệ', 'Thuyết minh', 'chỉ số']
    
    for sheet in finance_sheets:
        df = data[sheet]
        
        # Remove true blank header rows (e.g., '1. Hiệu quả hoạt động' where all Qs are NaN)
        # Check if all quarter columns are NaN for a row
        all_q_nan_idx = df[df.iloc[:, 1:].isnull().all(axis=1)].index
        df = df.drop(index=all_q_nan_idx).reset_index(drop=True)
        
        # Fill remaining isolated NaNs with 0 (since it means not generated/allocated)
        df.iloc[:, 1:] = df.iloc[:, 1:].fillna(0)
        
        data[sheet] = df
        
    # 2. Price Preprocessing
    price_df = data['Giá']
    
    # Highlight Outliers in Block Trades (Khối lượng Thỏa thuận)
    # Define an outlier threshold, e.g., 99th percentile or standard deviation based.
    # We will use > 99th percentile as "đột biến" (abnormal).
    if 'KL THỎA THUẬN' in price_df.columns:
        threshold = price_df['KL THỎA THUẬN'].quantile(0.99)
        # Create a dummy flag column to highlight the trade
        price_df['OUTLIER_BLOCK_TRADE'] = (price_df['KL THỎA THUẬN'] >= threshold).astype(int)
        
    data['Giá'] = price_df
    
    return data
