import pandas as pd
import numpy as np


def load_raw_with_audit(file_path):
    """
    Load data.xlsx, extract audit status rows BEFORE dropping them,
    then clean and slice as usual.
    Returns (data_dict, audit_dict).
      - data_dict: same as load_and_slice_financials output
      - audit_dict: {sheet_name: pd.Series} with audit status per quarter
    """
    xls = pd.ExcelFile(file_path)
    finance_sheets = ['Bảng cân đối kế toán', 'Kết quả kinh doanh',
                      'Lưu chuyển tiền tệ', 'Thuyết minh', 'chỉ số']

    audit_dict = {}
    data = {}

    for sheet in finance_sheets:
        df = pd.read_excel(xls, sheet_name=sheet)

        # Extract audit row before dropping
        audit_row_idx = df[df.iloc[:, 0] == 'Trạng thái kiểm toán'].index
        if len(audit_row_idx) > 0:
            audit_series = df.loc[audit_row_idx[0], df.columns[1:]]
            audit_dict[sheet] = audit_series
            df = df.drop(index=audit_row_idx).reset_index(drop=True)

        # Slice from Q1/2010 onwards
        cols = list(df.columns)
        try:
            start_idx = cols.index('Q1/2010')
            keep_cols = [cols[0]] + cols[start_idx:]
            df = df[keep_cols]
        except ValueError:
            pass

        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        data[sheet] = df

    # Price data
    price_df = pd.read_excel(xls, sheet_name='dữ liệu giá')
    data['Giá'] = price_df

    return data, audit_dict


def load_and_slice_financials(file_path):
    """
    Load data.xlsx, process financial sheets to drop audit status, 
    and slice data from Q1/2010 onwards.
    """
    xls = pd.ExcelFile(file_path)
    finance_sheets = ['Bảng cân đối kế toán', 'Kết quả kinh doanh', 'Lưu chuyển tiền tệ', 'Thuyết minh', 'chỉ số']
    
    data = {}
    
    for sheet in finance_sheets:
        df = pd.read_excel(xls, sheet_name=sheet)
        
        # 1. Drop audit status row if exists (usually at index 0)
        audit_row_idx = df[df.iloc[:, 0] == 'Trạng thái kiểm toán'].index
        if len(audit_row_idx) > 0:
            df = df.drop(index=audit_row_idx).reset_index(drop=True)
            
        # 2. Identify columns to keep (Chỉ tiêu + Q1/2010 onwards)
        # Convert column names to list
        cols = list(df.columns)
        
        # Find index of Q1/2010
        try:
            start_idx = cols.index('Q1/2010')
            # Keep the first column (Chỉ tiêu) and all columns from Q1/2010 onwards
            keep_cols = [cols[0]] + cols[start_idx:]
            df = df[keep_cols]
        except ValueError:
            print(f"Warning: 'Q1/2010' not found in {sheet}. Keeping all columns.")
            
        # Convert all quarter columns to float to ensure correct datatype
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        data[sheet] = df
        
    # Load price data separately (no column slicing needed)
    price_df = pd.read_excel(xls, sheet_name='dữ liệu giá')
    data['Giá'] = price_df
    
    return data

if __name__ == "__main__":
    # Test loader
    data = load_and_slice_financials(r'c:\fiinpro\data.xlsx')
    for name, df in data.items():
        print(f"[{name}] Shape: {df.shape}")
