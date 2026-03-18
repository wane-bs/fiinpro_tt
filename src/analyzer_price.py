import pandas as pd
import numpy as np
import warnings

def analyze_price_dynamics(price_df, output_dir):
    """
    Luồng 1: Phân tích Dữ liệu Giá (Price Dynamics).
    Tính toán Volatility và hiển thị thông tin thống kê.
    """
    print("\n--- LUỒNG 1: PHÂN TÍCH DỮ LIỆU GIÁ ---")
    
    # Sort by date
    if 'NGÀY' in price_df.columns:
        price_df = price_df.sort_values(by='NGÀY').reset_index(drop=True)
    
    # Calculate Daily Return
    # Assuming 'GIÁ' is the closing price
    if 'GIÁ' in price_df.columns:
        # Calculate daily log return to normalize
        price_df['LOG_RETURN'] = np.log(price_df['GIÁ'] / price_df['GIÁ'].shift(1))
        
        # Calculate trailing 30-day volatility (annualized)
        # 252 trading days in a year
        price_df['VOLATILITY_30D'] = price_df['LOG_RETURN'].rolling(window=30).std() * np.sqrt(252)
        
    # Analyze Block Trades
    if 'OUTLIER_BLOCK_TRADE' in price_df.columns:
        abnormal_days = price_df[price_df['OUTLIER_BLOCK_TRADE'] == 1]
        print(f"Phát hiện {len(abnormal_days)} phiên giao dịch có khối lượng thỏa thuận đột biến (>99th percentile).")
        
    # Analyze Seasonality (Hypothesis Testing)
    if 'NGÀY' in price_df.columns and 'LOG_RETURN' in price_df.columns:
        price_df['MONTH'] = price_df['NGÀY'].dt.month
        monthly_stats = price_df.groupby('MONTH').agg(
            Avg_Log_Return=('LOG_RETURN', 'mean'),
            Avg_Volatility=('VOLATILITY_30D', 'mean')
        ).reset_index()
        
        # Multiply by 100 for percentage
        monthly_stats['Avg_Log_Return_Pct'] = monthly_stats['Avg_Log_Return'] * 100
        
        # Generate Hypothesis Insight for April-May
        apr_may = monthly_stats[monthly_stats['MONTH'].isin([4, 5])]['Avg_Log_Return_Pct'].mean()
        other_months = monthly_stats[~monthly_stats['MONTH'].isin([4, 5])]['Avg_Log_Return_Pct'].mean()
        
        if apr_may > other_months:
            insight = f"**Chấp nhận Giả thuyết:** Tính thời vụ tháng 4-5 thể hiện rõ ràng. Trung bình tăng trưởng tháng 4-5 ({apr_may:.4f}%) **cao hơn** so với trung bình các tháng còn lại trong năm ({other_months:.4f}%)."
        else:
            insight = f"**Bác bỏ Giả thuyết:** Trung bình tăng trưởng tháng 4-5 ({apr_may:.4f}%) **thấp hơn hoặc bằng** so với trung bình các tháng còn lại ({other_months:.4f}%). Không có dấu hiệu bùng nổ rõ rệt vào quý 2."
            
        # Save insight as well
        with open(f"{output_dir}/price_seasonality_insight.txt", "w", encoding='utf-8') as f:
            f.write(insight)
            
        out_seasonality = f"{output_dir}/price_seasonality.csv"
        monthly_stats.to_csv(out_seasonality, index=False)
        print(f"Đã lưu kết quả kiểm định mùa vụ tại: {out_seasonality}")
        
    # Save the processed price data with features
    out_file = f"{output_dir}/price_analysis_features.csv"
    price_df.to_csv(out_file, index=False)
    print(f"Đã lưu đặc trưng dữ liệu giá tại: {out_file}")
    
    return price_df
