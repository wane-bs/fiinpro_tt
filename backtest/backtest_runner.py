"""
backtest_runner.py
==================
Entry point để chạy toàn bộ Backtest Suite.

Sử dụng:
  cd src
  python backtest_runner.py           # dùng output/ mặc định
  python backtest_runner.py --help
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

from backtest_seasonality import run_seasonality_backtest
from backtest_oos import run_oos_validation
from backtest_revenue_signal import run_revenue_signal_backtest
import json


THRESHOLDS = {
    # Ngưỡng chấp nhận / bác bỏ từng kiểm định
    'seasonality_hit_rate_min': 0.60,
    'trading_win_rate_min': 0.55,
    'trading_sharpe_min': 0.50,
    'oos_r2_min': {'Tổng Tài sản': 0.90, 'Tổng Nguồn vốn': 0.90, 'Tổng Doanh thu': 0.85},
    'oos_mape_max': {'Tổng Tài sản': 5.0, 'Tổng Nguồn vốn': 5.0, 'Tổng Doanh thu': 8.0},
    'revenue_hit_rate_min': 0.55,
    'revenue_alpha_min_pct': 2.0,
}


def _verdict_icon(passed: bool) -> str:
    return "✅" if passed else "❌"


def generate_backtest_report(
    seasonality_res: dict,
    oos_res: dict,
    revenue_res: dict,
    output_dir: str
) -> str:
    """Tổng hợp kết quả thành Markdown report."""

    lines = []
    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    lines.append(f"# Báo cáo Kiểm định Ngược (Backtest Report)\n")
    lines.append(f"> Được tạo tự động lúc: {now}\n")
    lines.append(f"> Train cutoff: **Q4/2021** | Test set: **Q1/2022 → Q4/2025**\n")
    lines.append("---\n")

    # ── 1. OOS Validation ─────────────────────────────────────────
    lines.append("## 1. OOS Validation — Mô hình RandomForest\n")
    lines.append("| Target | R²IS | R²OOS | MAPE_OOS (%) | RMSE Ratio | Verdict |")
    lines.append("|---|---|---|---|---|---|")
    oos_df = oos_res.get('summary', pd.DataFrame())
    overall_oos_pass = True
    for _, row in oos_df.iterrows():
        if 'error' in row:
            lines.append(f"| {row.get('target','')} | — | — | — | — | ⚠️ {row.get('error','')} |")
            continue
        t = row.get('target', '')
        verdict = row.get('Verdict', '')
        passed = '✅' in verdict
        if not passed:
            overall_oos_pass = False
        lines.append(f"| {t} | {row.get('R2_IS','?')} | {row.get('R2_OOS','?')} | "
                     f"{row.get('MAPE_OOS_Pct','?')} | {row.get('RMSE_OOS_vs_IS_Ratio','?')} | {verdict} |")

    lines.append("")
    lines.append("**Nhận xét:**")
    if overall_oos_pass:
        lines.append("- Mô hình RandomForest **không có dấu hiệu overfitting nghiêm trọng**. "
                     "Kết quả OOS hợp lệ — Feature Importance đáng tin cậy.")
    else:
        lines.append("- ⚠️ **Cảnh báo:** Một hoặc nhiều mô hình không đạt ngưỡng OOS. "
                     "Nên xem xét lại Feature Selection hoặc tăng Train set.")
    lines.append("")

    # ── 2. Walk-Forward Seasonality ───────────────────────────────
    lines.append("## 2. Walk-Forward Test — Mùa vụ Giá Tháng 4–5\n")
    hr = seasonality_res.get('hit_rate', 0)
    hr_pass = hr >= THRESHOLDS['seasonality_hit_rate_min']
    lines.append(f"| Chỉ tiêu | Kết quả | Ngưỡng | Đánh giá |")
    lines.append(f"|---|---|---|---|")
    lines.append(f"| Walk-Forward Hit Rate | {hr*100:.1f}% | ≥{THRESHOLDS['seasonality_hit_rate_min']*100:.0f}% | {_verdict_icon(hr_pass)} |")

    wr = seasonality_res.get('win_rate', 0)
    wr_pass = wr >= THRESHOLDS['trading_win_rate_min']
    lines.append(f"| Trading Win Rate (có phí) | {wr*100:.1f}% | ≥{THRESHOLDS['trading_win_rate_min']*100:.0f}% | {_verdict_icon(wr_pass)} |")

    sh = seasonality_res.get('sharpe', 0)
    sh_pass = sh >= THRESHOLDS['trading_sharpe_min']
    lines.append(f"| Sharpe Ratio | {sh:.3f} | ≥{THRESHOLDS['trading_sharpe_min']} | {_verdict_icon(sh_pass)} |")

    avg_r = seasonality_res.get('avg_net_return', 0)
    avg_a = seasonality_res.get('avg_alpha', 0)
    lines.append(f"| Avg Net Return T4–T5 | {avg_r:.2f}% | — | — |")
    lines.append(f"| Avg Alpha vs Buy&Hold | {avg_a:.2f}% | — | — |")
    lines.append("")

    lines.append("**Nhận xét:**")
    if hr_pass and wr_pass:
        lines.append("- Giả thuyết mùa vụ tháng 4–5 **được kiểm định xác nhận**. "
                     "Quy tắc giao dịch có cơ sở thực nghiệm.")
    elif hr_pass or wr_pass:
        lines.append("- Giả thuyết mùa vụ **đạt 1/2 tiêu chí**. Tín hiệu có nhưng không đủ mạnh để "
                     "giao dịch cơ học — cần kết hợp xác nhận kỹ thuật thêm.")
    else:
        lines.append("- ❌ **Bác bỏ giả thuyết mùa vụ.** Hit rate và Win rate đều dưới ngưỡng. "
                     "Không nên đặt lệnh thuần túy dựa trên mùa vụ tháng 4–5.")

    # Chi tiết từng năm
    wf_df = seasonality_res.get('walkforward', pd.DataFrame())
    if not wf_df.empty:
        lines.append("\n**Walk-Forward Detail (từng năm):**\n")
        lines.append("| Năm | Dự báo Dương | Thực tế Dương | T4-T5 Return (%) | Hit |")
        lines.append("|---|---|---|---|---|")
        for _, r in wf_df.iterrows():
            hit_icon = "✅" if r['Hit'] else "❌"
            lines.append(f"| {int(r['Year'])} | {'✅' if r['Prediction_AprMay_Positive'] else '❌'} | "
                         f"{'✅' if r['Actual_Positive'] else '❌'} | "
                         f"{r['Actual_AprMay_AvgReturn_Pct']:.3f}% | {hit_icon} |")
    lines.append("")

    # ── 3. Revenue Signal ─────────────────────────────────────────
    lines.append("## 3. Tín hiệu Doanh thu → Forward Return\n")
    rev_summary = revenue_res.get('summary', pd.DataFrame())
    if not rev_summary.empty:
        lines.append("| Ngưỡng QoQ | N Signal | Hit Rate | Avg Fwd Return | Alpha vs Base | Verdict |")
        lines.append("|---|---|---|---|---|---|")
        for _, row in rev_summary.iterrows():
            lines.append(f"| >{row['Threshold_Pct']:.0f}% | {row['signal_n']} | "
                         f"{row['signal_hit_rate']*100:.1f}% | "
                         f"{row['signal_avg_fwd_return_pct']:.2f}% | "
                         f"{row['alpha_vs_base_pct']:.2f}% | {row['Verdict']} |")

    lines.append("")
    lines.append("**Nhận xét:**")
    best_row = rev_summary[rev_summary['Verdict'].str.contains('✅')].head(1)
    if not best_row.empty:
        br = best_row.iloc[0]
        lines.append(f"- Tín hiệu tốt nhất ở ngưỡng **QoQ >{br['Threshold_Pct']:.0f}%**: "
                     f"Hit Rate {br['signal_hit_rate']*100:.1f}%, Alpha {br['alpha_vs_base_pct']:.2f}%. "
                     f"Doanh thu tăng trưởng mạnh là chỉ báo sớm có giá trị.")
    else:
        lines.append("- Tín hiệu doanh thu chưa đủ mạnh ở các ngưỡng kiểm tra. "
                     "Cần kết hợp thêm yếu tố khác (ví dụ: volatility thấp, KL tăng).")
    lines.append("")

    # ── 4. Ma trận Bác bỏ ────────────────────────────────────────
    lines.append("## 4. Ma trận Bác bỏ Luận điểm (Cập nhật sau backtest)\n")
    lines.append("| Luận điểm | Điều kiện Bác bỏ | Trạng thái Hiện tại |")
    lines.append("|---|---|---|")

    seas_ok = "Còn giá trị" if hr_pass else "⚠️ Cần xem lại"
    oos_ok = "Mô hình ổn định" if overall_oos_pass else "⚠️ Nghi ngờ overfitting"
    lines.append(f"| Tháng 4–5 mạnh hơn | Hit rate < 50% | {seas_ok} |")
    lines.append(f"| RF Feature Importance ổn định | R²OOS < 0.7 | {oos_ok} |")
    lines.append(f"| KL tăng = tích lũy | Giá tiếp tục giảm sau Q1/2026 dù KL cao | Chờ xác nhận |")
    lines.append(f"| Doanh thu kỷ lục hỗ trợ giá | DT Q1/2026 giảm >20% QoQ | Chờ BCTC |")
    lines.append(f"| Vốn chủ mạnh = solvency | D/E tăng đột biến | Chờ BCTC |")
    lines.append("")

    # ── 5. Khuyến nghị Định lượng (Composite Signal) ──────────────
    lines.append("## 5. Định giá & Khuyến nghị (Trụ cột 3 & 4)\n")
    rec_path = os.path.join(output_dir, 'recommendation.json')
    if os.path.exists(rec_path):
        try:
            with open(rec_path, 'r', encoding='utf-8') as f:
                rec_data = json.load(f)
            
            score = rec_data.get('Composite_Score', 0)
            verdict = rec_data.get('Verdict', 'N/A')
            lines.append(f"### Mức độ Bứt phá (Composite Score): **{score:+.1f} / 100**")
            lines.append(f"**Khuyến nghị Hiện tại:** 🎯 **{verdict}**\n")
            
            lines.append("**Cấu thành Điểm số:**")
            comps = rec_data.get('Components', {})
            for k, v in comps.items():
                lines.append(f"- {k}: {v:+.1f}")
            lines.append("")
        except Exception as e:
            lines.append(f"*(Lỗi đọc recommendation.json: {e})*\n")
    else:
        lines.append("*(Chưa có dữ liệu Signal Engine)*\n")

    target_path = os.path.join(output_dir, 'target_price_scenarios.csv')
    if os.path.exists(target_path):
        lines.append("### Khung Định giá Target Price (1 năm tới)")
        try:
            target_df = pd.read_csv(target_path)
            lines.append("| Kịch bản | Target Price (VNĐ) | Upside (%) |")
            lines.append("|---|---|---|")
            for _, r in target_df.iterrows():
                icon = "📈" if r['Upside_Pct'] > 0 else "📉"
                lines.append(f"| {icon} {r['Scenario']} | {r['Avg_Target']:,.0f} | {r['Upside_Pct']:+.1f}% |")
            lines.append("")
        except:
            pass

    # ── 6. Kết luận tổng thể ──────────────────────────────────────
    lines.append("## 6. Kết luận Tổng thể Của Mô Hình Cũ\n")
    passed_count = sum([overall_oos_pass, hr_pass and wr_pass])
    if passed_count == 2:
        conclusion = "🟢 **MÔ HÌNH VÀ LUẬN ĐIỂM ĐỀU VỮNG** — Khuyến nghị tích lũy có cơ sở thực nghiệm."
    elif passed_count == 1:
        conclusion = "🟡 **MỘT PHẦN ĐẠT** — Tiếp tục theo dõi, chưa đủ điều kiện commit đầy đủ."
    else:
        conclusion = "🔴 **CẦN ĐIỀU CHỈNH** — Xem lại mô hình và giả thuyết trước khi đặt lệnh."

    lines.append(conclusion)

    report_md = "\n".join(lines)
    report_path = os.path.join(output_dir, "backtest_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"\n  📄 Đã tạo báo cáo tổng hợp: {report_path}")
    return report_md


def main():
    parser = argparse.ArgumentParser(description="Chạy Backtest Suite phân tích FiinPro")
    parser.add_argument("--output-dir", default=None,
                        help="Thư mục output chứa CSV (mặc định: ../output/)")
    parser.add_argument("--skip-oos", action="store_true", help="Bỏ qua OOS validation")
    parser.add_argument("--skip-seasonality", action="store_true", help="Bỏ qua kiểm định mùa vụ")
    parser.add_argument("--skip-revenue", action="store_true", help="Bỏ qua backtest doanh thu")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, "output")
    print(f"=== BACKTEST SUITE === Output dir: {output_dir}")

    seasonality_res = {}
    oos_res = {}
    revenue_res = {}

    if not args.skip_seasonality:
        try:
            seasonality_res = run_seasonality_backtest(output_dir)
        except Exception as e:
            print(f"[ERROR] Backtest mùa vụ: {e}")

    if not args.skip_oos:
        try:
            oos_res = run_oos_validation(output_dir)
        except Exception as e:
            print(f"[ERROR] OOS validation: {e}")

    if not args.skip_revenue:
        try:
            revenue_res = run_revenue_signal_backtest(output_dir)
        except Exception as e:
            print(f"[ERROR] Backtest doanh thu: {e}")

    generate_backtest_report(seasonality_res, oos_res, revenue_res, output_dir)
    print("\n=== BACKTEST HOÀN THÀNH ===")


if __name__ == "__main__":
    main()
