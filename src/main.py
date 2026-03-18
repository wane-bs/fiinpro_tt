import sys
import os
import shutil

# Resolve paths relative to this script's location so the project is portable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

from data_loader import load_raw_with_audit
from preprocessor import preprocess_data
from analyzer_preanalysis import run_preanalysis
from analyzer_price import analyze_price_dynamics
from analyzer_fin import analyze_financials
from analyzer_cross import run_causal_analysis
from analyzer_structure import run_structure_analysis
from analyzer_cycle import run_cycle_analysis
from analyzer_ratios import run_ratio_analysis, run_vertical_analysis
from analyzer_dupont import run_dupont_analysis
from analyzer_cashflow import run_cashflow_analysis
from analyzer_dcf import run_dcf_analysis
from analyzer_valuation import run_valuation_analysis, run_multiples_valuation
from signal_engine import generate_composite_signal

def main():
    # Accept data file path as optional CLI argument; default to data.xlsx in project root
    if len(sys.argv) > 1:
        source_file = sys.argv[1]
    else:
        source_file = os.path.join(PROJECT_ROOT, 'data.xlsx')
    output_dir = os.path.join(PROJECT_ROOT, 'output')
    
    # 0. Clean output directory
    if os.path.exists(output_dir):
        print(f"Cleaning output directory: {output_dir}")
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.is_dir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        os.makedirs(output_dir)
        
    print(f"Loading data from {source_file}...")
    try:
        raw_data, audit_dict = load_raw_with_audit(source_file)
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        sys.exit(1)
        
    print(f"\nPreprocessing data (Cleaning NaNs and Nulls)...")
    clean_data = preprocess_data(raw_data)
    
    # Module 1.1: Pre-Analysis (Data Quality & Audit)
    run_preanalysis(clean_data, audit_dict, output_dir)
    
    # Run luồng 1
    processed_price = analyze_price_dynamics(clean_data['Giá'], output_dir)
    
    # Run luồng 2
    analyze_financials(clean_data, output_dir)
    
    # Run luồng 3
    run_causal_analysis(clean_data, processed_price, output_dir)
    
    # Run luồng 4 (Kiểm định & Phân tích cấu trúc RF)
    run_structure_analysis(clean_data, output_dir)
    
    # Luồng 5b: Phân tích Chu kỳ (STL decomposition + Cross-Correlation)
    run_cycle_analysis(clean_data, output_dir)
    
    # Module 1.3: Vertical Analysis
    run_vertical_analysis(clean_data, output_dir)
    
    # Run luồng 5 (Financial Ratios — includes Module 1.4 + 2.2)
    run_ratio_analysis(clean_data, output_dir)
    
    # Module 2.3: DuPont 3-Step Analysis
    run_dupont_analysis(clean_data, output_dir)
    
    # Module 2.4: Cash Flow Quality
    run_cashflow_analysis(clean_data, output_dir)
    
    # Module 3.2: DCF Valuation + Sensitivity
    run_dcf_analysis(clean_data, output_dir)
    
    # Run luồng 6 (Valuation — P/S, Log-Reg, Mean Reversion)
    run_valuation_analysis(output_dir)
    
    # Module 3.3: Multi-Multiple Valuation (P/E, P/B, EV/EBITDA)
    run_multiples_valuation(clean_data, output_dir)
    
    # Run luồng 7 (Composite Signal)
    generate_composite_signal(output_dir)
    
    print("\nChương trình pipeline phân tích kết hợp định lượng đã kết thúc.")
    print(f"Kiểm tra thư mục {output_dir} để xem báo cáo và tín hiệu đầu tư.")

if __name__ == "__main__":
    main()
