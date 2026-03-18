import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os

st.set_page_config(page_title="Fiinpro Analysis Dashboard", layout="wide")

st.title("📊 Fiinpro Analysis Dashboard")
st.markdown("Hệ thống phân tích tài chính định lượng — Cấu trúc & Chu kỳ")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=0)
def load_data():
    base = os.path.join(_PROJECT_ROOT, "output")

    def _csv(name, **kwargs):
        try:
            return pd.read_csv(os.path.join(base, name), **kwargs)
        except Exception:
            return pd.DataFrame()

    def _txt(name):
        try:
            with open(os.path.join(base, name), "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""

    def _json(name):
        try:
            with open(os.path.join(base, name), "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    return {
        # Chương 1
        "preanalysis": _json("preanalysis_report.json"),
        "vertical":    _csv("vertical_analysis.csv"),
        # Chương 2
        "dupont":      _csv("dupont_analysis.csv"),
        "cashflow":    _csv("cashflow_quality.csv"),
        "cycle_decomp":_csv("cycle_decomposition.csv"),
        # Chương 3
        "val_bands":   _csv("valuation_bands.csv"),
        "dcf_sens":    _csv("dcf_sensitivity.csv"),
        "target_price":_csv("target_price_scenarios.csv"),
        "dcf_val":     _csv("dcf_valuation.csv"),
        "multiples":   _csv("multiples_valuation.csv"),
        # Chương 4
        "cycle_ccf":   _csv("cycle_cross_correlation.csv"),
        "impact":      _csv("structure_impact.csv"),
        "metrics":     _csv("model_metrics.csv"),
        "ts":          _csv("structure_timeseries.csv"),
        # Báo cáo
        "signal":      _csv("composite_signal.csv"),
        "rec":         _json("recommendation.json"),
        "report_md":   _txt("analysis_report.md"),
        "cycle_md":    _txt("cycle_report.md"),
        # Extra
        "price":       _csv("price_analysis_features.csv"),
        "revenue":     _csv("revenue_growth.csv"),
        "ratios":      _csv("financial_ratios.csv"),
    }

d = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📘 Chương 1 — Sức khỏe Nền tảng",
    "📗 Chương 2 — Hiệu suất & Chu kỳ",
    "📙 Chương 3 — Định giá & Độ nhạy",
    "📕 Chương 4 — Cấu trúc & Chu kỳ",
    "📋 Báo cáo — Luận điểm & Kết luận",
])

# ═════════════════════════════════════════════════════════════════════════════
# CHƯƠNG 1: Sức khỏe Tài chính Nền tảng
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Chương 1: Phân tích Nền tảng & Sức khỏe Tài chính")
    st.markdown("**Trọng tâm:** Xác lập độ tin cậy dữ liệu và hình thái tài chính tĩnh.")

    # ── 1.1: Audit Reliability Gauge ──────────────────────────────────────
    st.divider()
    st.subheader("1.1 Sơ đồ Trạng thái Kiểm toán (Audit Reliability Gauge)")
    st.markdown("*Phương pháp: Thống kê định tính — Tỷ lệ = (Số quý Audited / Tổng số quý) × 100*")

    pre = d["preanalysis"]
    if pre:
        audit_rate = pre.get("audit_rate_pct", 0)
        total_q    = pre.get("total_quarters", 0)
        audited_q  = pre.get("audited_quarters", 0)
        flag       = pre.get("data_quality_flag", "N/A")
        missing    = pre.get("missing_rate_pct", 0)
        max_consec = pre.get("max_consecutive_unaudited", 0)

        c1, c2, c3, c4 = st.columns(4)
        flag_color = {"OK": "normal", "CAUTION": "off", "WARNING": "inverse"}.get(flag, "normal")
        c1.metric("Tỷ lệ Kiểm toán", f"{audit_rate:.1f}%", f"{audited_q}/{total_q} quý")
        c2.metric("Trạng thái Dữ liệu", flag)
        c3.metric("Missing Rate", f"{missing:.2f}%")
        c4.metric("Max Quý chưa kiểm toán LT", f"{max_consec} quý")

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=audit_rate,
            title={"text": "Tỷ lệ Kiểm toán (%)"},
            delta={"reference": 50},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2196F3"},
                "steps": [
                    {"range": [0, 20],  "color": "#FFEBEE"},
                    {"range": [20, 50], "color": "#FFF3E0"},
                    {"range": [50, 100],"color": "#E8F5E9"},
                ],
                "threshold": {"line": {"color": "red", "width": 3}, "thickness": 0.75, "value": 50}
            }
        ))
        fig_gauge.update_layout(height=280)
        st.plotly_chart(fig_gauge, use_container_width=True, key="audit_gauge")

        policy = pre.get("policy_keywords_found", [])
        if policy:
            with st.expander(f"📜 Chính sách Kế toán phát hiện ({len(policy)} dòng)"):
                for row in policy:
                    st.markdown(f"- {row}")
    else:
        st.info("Chưa có dữ liệu kiểm toán. Hãy chạy `python main.py` để tạo.")

    # ── 1.2: Common-size Stacked Bar ──────────────────────────────────────
    st.divider()
    st.subheader("1.2 Sơ đồ Cơ cấu Tài sản & Nguồn vốn (Common-size 100% Stacked Bar)")
    st.markdown("*Phương pháp: Common-size Analysis — Tỷ trọng = (Khoản mục con / Tổng) × 100*")

    vert = d["vertical"]
    if not vert.empty:
        quarter_cols = [c for c in vert.columns if c not in ["Chỉ_tiêu", "Báo_cáo", "Mẫu_số"]]

        for bao_cao_label, mau_so_label, title in [
            ("BCĐKT", "Tổng tài sản",   "Cơ cấu Tài sản (% Tổng tài sản)"),
            ("BCĐKT", "Tổng nguồn vốn", "Cơ cấu Nguồn vốn (% Tổng nguồn vốn)"),
            ("KQKD",  "Doanh thu thuần","Cơ cấu Kết quả Kinh doanh (% Doanh thu thuần)"),
        ]:
            sub = vert[(vert["Báo_cáo"] == bao_cao_label) & (vert["Mẫu_số"] == mau_so_label)].copy()
            if sub.empty:
                continue

            # Melt sang long format
            melt = sub.melt(id_vars=["Chỉ_tiêu"], value_vars=quarter_cols,
                            var_name="Quý", value_name="Tỷ trọng (%)")
            fig_stack = px.bar(
                melt, x="Quý", y="Tỷ trọng (%)", color="Chỉ_tiêu",
                title=title, barmode="relative",
                labels={"Tỷ trọng (%)": "Tỷ trọng (%)", "Quý": "Quý"},
                text_auto=".1f"
            )
            fig_stack.update_layout(height=420, legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig_stack, use_container_width=True, key=f"stacked_{mau_so_label}")

        # ── 1.2b: Bảng Đối chiếu First vs Last + Pie Chart Mô hình ──────
        st.divider()
        st.subheader("📊 Bảng Đối chiếu Cơ cấu & Biểu đồ Tròn Mô hình")
        st.markdown("*Biểu đồ tròn thể hiện tỷ trọng trung bình lịch sử (mean toàn bộ quý) — tổng có thể ≠ 100%*")

        for mau_so, title_pie in [
            ("Tổng tài sản",   "Cơ cấu Tài sản — Tỷ trọng TB Lịch sử"),
            ("Tổng nguồn vốn", "Cơ cấu Nguồn vốn — Tỷ trọng TB Lịch sử"),
        ]:
            sub_pie = vert[(vert["Báo_cáo"] == "BCĐKT") & (vert["Mẫu_số"] == mau_so)].copy()
            if sub_pie.empty:
                continue

            first_q = quarter_cols[0]
            last_q = quarter_cols[-1]
            sub_pie_numeric = sub_pie[quarter_cols].astype(float)

            # Bảng đối chiếu
            compare_df = pd.DataFrame({
                "Khoản mục": sub_pie["Chỉ_tiêu"].values,
                f"{first_q} (%)": sub_pie_numeric[first_q].values,
                f"{last_q} (%)": sub_pie_numeric[last_q].values,
            })
            compare_df["Biến động (+/-)"] = compare_df[f"{last_q} (%)"] - compare_df[f"{first_q} (%)"]
            compare_df["Biến động (+/-)"] = compare_df["Biến động (+/-)"].round(2)

            col_table, col_pie = st.columns([1, 1])
            with col_table:
                st.markdown(f"**{mau_so} — {first_q} vs {last_q}:**")
                st.dataframe(compare_df, hide_index=True, use_container_width=True)
            with col_pie:
                # Tỷ trọng trung bình lịch sử
                avg_values = sub_pie_numeric.mean(axis=1).values
                labels = sub_pie["Chỉ_tiêu"].values
                fig_pie = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=np.abs(avg_values),
                    textinfo="label+percent",
                    hovertemplate="<b>%{label}</b><br>TB Lịch sử: %{value:.1f}%<extra></extra>",
                )])
                fig_pie.update_layout(title=title_pie, height=360, showlegend=True)
                st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{mau_so}")

    else:
        st.info("Chưa có dữ liệu Vertical Analysis. Hãy chạy `python main.py` để tạo.")

    # ── 1.3: Kết luận Tự động Loại hình Kinh doanh ───────────────────────
    st.divider()
    st.subheader("1.3 Kết luận Tự động Loại hình Kinh doanh")
    st.markdown("*Phương pháp: Rule-based Classification dựa trên chỉ số tài chính quý gần nhất*")

    ratios = d["ratios"]
    if not vert.empty and not ratios.empty:
        quarter_cols_v = [c for c in vert.columns if c not in ["Chỉ_tiêu", "Báo_cáo", "Mẫu_số"]]
        last_q_v = quarter_cols_v[-1] if quarter_cols_v else None
        last_ratios = ratios.iloc[-1]

        # Trích xuất chỉ số từ vertical analysis (quý cuối)
        def _get_vert_val(chi_tieu, mau_so):
            row = vert[(vert["Chỉ_tiêu"] == chi_tieu) & (vert["Mẫu_số"] == mau_so)]
            if not row.empty and last_q_v:
                return float(row[last_q_v].values[0])
            return 0.0

        tscd_pct = _get_vert_val("Tài sản cố định", "Tổng tài sản")
        tien_pct = _get_vert_val("Tiền và tương đương", "Tổng tài sản")
        asset_turnover = float(last_ratios.get("Asset_Turnover", 0))
        de_ratio = float(last_ratios.get("DE_Ratio", 0))

        conclusions = []
        if tscd_pct > 40:
            conclusions.append(("⚙️", "Mô hình Thâm dụng vốn", f"TSCĐ/Tổng TS = {tscd_pct:.1f}% > 40%"))
        if asset_turnover > 1.5:
            conclusions.append(("🔄", "Bán lẻ / Quay vòng nhanh", f"Vòng quay TS = {asset_turnover:.2f} > 1.5"))
        if de_ratio > 2.0:
            conclusions.append(("⚠️", "Sử dụng Đòn bẩy cao", f"Nợ/VCSH = {de_ratio:.2f} > 2.0"))
        if tien_pct > 20:
            conclusions.append(("💰", "Mô hình Giàu tiền mặt", f"Tiền/Tổng TS = {tien_pct:.1f}% > 20%"))

        if conclusions:
            for icon, label, detail in conclusions:
                st.success(f"{icon} **{label}** — {detail}")
        else:
            # Tạo kết luận tổng hợp
            summary_parts = []
            summary_parts.append(f"Nợ/VCSH = {de_ratio:.2f}")
            summary_parts.append(f"TSCĐ = {tscd_pct:.1f}%")
            summary_parts.append(f"Vòng quay TS = {asset_turnover:.2f}")
            st.info(
                f"📋 **Mô hình Đầu tư hỗn hợp** — Không thuộc phân loại đặc biệt nào. "
                f"{', '.join(summary_parts)}. "
                f"Hiệu suất khai thác tài sản {'thấp' if asset_turnover < 0.5 else 'trung bình'}."
            )
    else:
        st.info("Chưa có đủ dữ liệu để phân loại. Hãy chạy `python main.py` để tạo.")

    # ── 1.4: Sức khỏe Thanh khoản ────────────────────────────────────────
    st.divider()
    st.subheader("1.4 Sức khỏe Thanh khoản")
    st.markdown("*Phương pháp: Tỷ số thanh toán — Current Ratio = TSNH/NNH, Quick Ratio = (TSNH−HTK)/NNH*")

    if not ratios.empty:
        last_r = ratios.iloc[-1]
        current_ratio = float(last_r.get("Current_Ratio", 0))
        quick_ratio = float(last_r.get("Quick_Ratio", 0))
        last_qtr = last_r.get("Quarter", "N/A")

        c1, c2, c3 = st.columns(3)
        c1.metric("Quý", last_qtr)
        c2.metric("Thanh toán hiện hành", f"{current_ratio:.2f}",
                  delta="Tốt" if current_ratio >= 1.0 else "Cần cải thiện",
                  delta_color="normal" if current_ratio >= 1.0 else "inverse")
        c3.metric("Thanh toán nhanh", f"{quick_ratio:.2f}",
                  delta="Tốt" if quick_ratio >= 1.0 else "Cần cải thiện",
                  delta_color="normal" if quick_ratio >= 1.0 else "inverse")

        if current_ratio >= 1.0 and quick_ratio >= 1.0:
            st.success("✅ Sức khỏe thanh khoản ở mức **Tốt**, đảm bảo khả năng chi trả nghĩa vụ ngắn hạn.")
        elif current_ratio >= 1.0:
            st.warning("⚠️ Thanh toán hiện hành ổn nhưng Quick Ratio thấp — hàng tồn kho chiếm tỷ trọng lớn.")
        else:
            st.error("❌ Thanh khoản **yếu** — rủi ro không đủ tài sản ngắn hạn để thanh toán nợ đến hạn.")
    else:
        st.info("Chưa có dữ liệu Financial Ratios. Hãy chạy `python main.py` để tạo.")


# ═════════════════════════════════════════════════════════════════════════════
# CHƯƠNG 2: Hiệu suất & Đồng bộ Chu kỳ
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Chương 2: Đánh giá Hiệu suất & Đồng bộ Chu kỳ")
    st.markdown("**Trọng tâm:** Bóc tách động lực ROE và tìm sự tương quan giữa các chu kỳ nội tại.")

    # ── 2.1: DuPont Waterfall ─────────────────────────────────────────────
    st.divider()
    st.subheader("2.1 Phân rã Động lực DuPont (DuPont Waterfall Chart)")
    st.markdown("*Phương pháp: DuPont 3 nhân tố — ROE = Net Margin × Asset Turnover × Equity Multiplier*")

    dup = d["dupont"]
    if not dup.empty:
        # Line chart: 3 factors + ROE over time
        fig_dup = go.Figure()
        fig_dup.add_trace(go.Scatter(x=dup["Quarter"], y=dup["Net_Margin_Pct"],
                                     mode="lines+markers", name="Net Margin (%)"))
        fig_dup.add_trace(go.Scatter(x=dup["Quarter"], y=dup["Asset_Turnover"],
                                     mode="lines+markers", name="Asset Turnover (x)", yaxis="y2"))
        fig_dup.add_trace(go.Scatter(x=dup["Quarter"], y=dup["Equity_Multiplier"],
                                     mode="lines+markers", name="Equity Multiplier (x)", yaxis="y2"))
        fig_dup.add_trace(go.Bar(x=dup["Quarter"], y=dup["ROE_DuPont_Pct"],
                                  name="ROE DuPont (%)", opacity=0.4, marker_color="#2196F3"))
        fig_dup.update_layout(
            title="Diễn biến DuPont 3 Nhân tố theo Quý",
            yaxis=dict(title="% (Net Margin, ROE)"),
            yaxis2=dict(title="Lần (Turnover, Multiplier)", overlaying="y", side="right"),
            xaxis_title="Quý",
            legend=dict(orientation="h", y=-0.25),
            height=450,
        )
        st.plotly_chart(fig_dup, use_container_width=True, key="dupont_chart")

        # Delta ROE Waterfall
        dup_clean = dup.dropna(subset=["Delta_ROE"]).tail(12)
        if not dup_clean.empty:
            fig_wf_dup = go.Figure(go.Waterfall(
                orientation="v",
                measure=["relative"] * len(dup_clean),
                x=dup_clean["Quarter"],
                y=dup_clean["Delta_ROE"],
                text=[f"{v:+.2f}%" for v in dup_clean["Delta_ROE"]],
                textposition="outside",
                increasing={"marker": {"color": "#00C853"}},
                decreasing={"marker": {"color": "#D50000"}},
                connector={"line": {"color": "#555"}},
            ))
            fig_wf_dup.update_layout(
                title="Thay đổi ROE (ΔROEᵢ) từng quý — Waterfall",
                yaxis_title="% ROE thay đổi",
                height=380,
                showlegend=False,
            )
            st.plotly_chart(fig_wf_dup, use_container_width=True, key="dupont_waterfall")
    else:
        st.info("Chưa có dữ liệu DuPont. Hãy chạy `python main.py` để tạo.")

    # ── 2.x: DuPont What-if Calculator ────────────────────────────────────
    st.divider()
    st.subheader("2.x Mô hình Hồi quy Cấu trúc — Kịch bản tăng 10% ROE")
    st.markdown("*Phương pháp: Giữ nguyên các nhân tố khác, tính giá trị cần thiết để ROE tăng 10%*")

    if not dup.empty:
        last_dup = dup.dropna(subset=["ROE_DuPont_Pct"]).iloc[-1]
        roe_current = float(last_dup["ROE_DuPont_Pct"])
        nm_current = float(last_dup["Net_Margin_Pct"])
        at_current = float(last_dup["Asset_Turnover"])
        em_current = float(last_dup["Equity_Multiplier"])
        roe_target = roe_current * 1.10

        st.markdown(f"**ROE hiện tại:** {roe_current:.2f}% → **Mục tiêu (+10%):** {roe_target:.2f}%")

        col_s1, col_s2 = st.columns(2)

        with col_s1:
            st.markdown("#### Kịch bản 1: Cải thiện Hiệu suất")
            st.markdown("*Tăng Vòng quay Tài sản, giữ nguyên Biên LN và Đòn bẩy*")
            if nm_current != 0 and em_current != 0:
                at_needed = roe_target / (nm_current / 100 * em_current * 100)
                st.metric("Vòng quay TS cần đạt", f"{at_needed:.4f}",
                          delta=f"{at_needed - at_current:+.4f} so với hiện tại ({at_current:.4f})")
                st.caption("💡 Đẩy mạnh thu hồi Phải thu khách hàng và tối ưu hóa Hàng tồn kho.")
            else:
                st.warning("Không thể tính — Net Margin hoặc Equity Multiplier = 0.")

        with col_s2:
            st.markdown("#### Kịch bản 2: Cải thiện Biên lợi nhuận")
            st.markdown("*Tăng Net Margin, giữ nguyên Vòng quay và Đòn bẩy*")
            if at_current != 0 and em_current != 0:
                nm_needed = roe_target / (at_current * em_current)
                st.metric("Biên LN ròng cần đạt", f"{nm_needed:.2f}%",
                          delta=f"{nm_needed - nm_current:+.2f}% so với hiện tại ({nm_current:.2f}%)")
                st.caption("💡 Tiết giảm Chi phí quản lý doanh nghiệp (hệ số tác động ≈ −0.22).")
            else:
                st.warning("Không thể tính — Asset Turnover hoặc Equity Multiplier = 0.")

    # ── 2.2: Accrual Analysis Dual-Axis ───────────────────────────────────
    st.divider()
    st.subheader("2.2 Đối chiếu Lợi nhuận — Dòng tiền (Dual-Axis Combo Chart)")
    st.markdown("*Phương pháp: Accrual Analysis — Sloan Accrual Ratio = (NI − CFO) / Avg Assets*")

    cf = d["cashflow"]
    if not cf.empty:
        fig_cf = go.Figure()
        if "CFO" in cf.columns:
            fig_cf.add_trace(go.Bar(x=cf["Quarter"], y=cf["CFO"],
                                    name="CFO (Dòng tiền kinh doanh)", marker_color="#1565C0", opacity=0.7))
        if "FCF" in cf.columns:
            fig_cf.add_trace(go.Bar(x=cf["Quarter"], y=cf["FCF"],
                                    name="FCF (Dòng tiền tự do)", marker_color="#00ACC1", opacity=0.7))
        if "CFO_NI_Ratio" in cf.columns:
            fig_cf.add_trace(go.Scatter(x=cf["Quarter"], y=cf["CFO_NI_Ratio"],
                                         mode="lines+markers", name="CFO/NI Ratio",
                                         yaxis="y2", line=dict(color="#FF6F00", width=2.5)))
            fig_cf.add_shape(type="line", x0=cf["Quarter"].iloc[0], x1=cf["Quarter"].iloc[-1],
                             y0=1, y1=1, yref="y2",
                             line=dict(color="green", dash="dash", width=1.5))

        fig_cf.update_layout(
            title="CFO, FCF & Tỷ số CFO/NI theo Quý",
            yaxis=dict(title="Giá trị (tỷ VNĐ)"),
            yaxis2=dict(title="CFO/NI Ratio (lần)", overlaying="y", side="right"),
            xaxis_title="Quý",
            barmode="group",
            height=420,
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig_cf, use_container_width=True, key="cashflow_chart")

        if "Accrual_Ratio" in cf.columns:
            fig_acr = px.bar(cf, x="Quarter", y="Accrual_Ratio",
                             title="Accrual Ratio (Sloan) theo Quý — Chỉ báo Lợi nhuận Ảo",
                             color="Accrual_Ratio",
                             color_continuous_scale=["#1B5E20", "#FFEB3B", "#B71C1C"],
                             color_continuous_midpoint=0)
            fig_acr.add_hline(y=0.05, line_dash="dash", line_color="red",
                               annotation_text="Ngưỡng cảnh báo 5%")
            fig_acr.update_layout(height=320, showlegend=False)
            st.plotly_chart(fig_acr, use_container_width=True, key="accrual_chart")
    else:
        st.info("Chưa có dữ liệu Cash Flow Quality. Hãy chạy `python main.py` để tạo.")

    # ── 2.3: Cycle Alignment (Z-score Overlay) ────────────────────────────
    st.divider()
    st.subheader("2.3 Đồng bộ hóa Chu kỳ (Cycle Alignment Plot)")
    st.markdown("*Phương pháp: Chuẩn hóa Z-score thành phần Trend (STL) → Chồng lớp quan sát lệch pha*")

    decomp = d["cycle_decomp"]
    if not decomp.empty:
        metrics_list = decomp["Chỉ số"].unique().tolist()
        fig_align = go.Figure()
        colors = ["#1565C0", "#B71C1C", "#2E7D32", "#6A1B9A"]
        for i, metric in enumerate(metrics_list):
            sub = decomp[decomp["Chỉ số"] == metric].copy()
            trend_vals = sub["Trend"].values.astype(float)
            mean_, std_ = np.nanmean(trend_vals), np.nanstd(trend_vals)
            z = (trend_vals - mean_) / (std_ if std_ > 0 else 1)
            fig_align.add_trace(go.Scatter(
                x=sub["Quý"], y=z,
                mode="lines", name=metric,
                line=dict(color=colors[i % len(colors)], width=2.5)
            ))
        fig_align.add_hline(y=0, line_dash="dash", line_color="#555", line_width=1)
        fig_align.update_layout(
            title="Cycle Alignment — Z-score Trend các Chỉ số Hiệu quả",
            yaxis_title="Z-score (Trend)", xaxis_title="Quý",
            legend=dict(orientation="h", y=-0.2),
            height=400,
        )
        st.plotly_chart(fig_align, use_container_width=True, key="cycle_align")
    else:
        st.info("Chưa có dữ liệu Cycle Decomposition. Hãy chạy `python main.py` để tạo.")


# ═════════════════════════════════════════════════════════════════════════════
# CHƯƠNG 3: Định giá & Độ nhạy
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Chương 3: Mô hình Định giá & Độ nhạy")
    st.markdown("**Trọng tâm:** Xác định vùng giá trị và rủi ro giả định.")

    # ── 3.1: Valuation Bands ──────────────────────────────────────────────
    st.divider()
    st.subheader("3.1 Dải Định giá Lịch sử (Valuation Bands Chart)")
    st.markdown("*Phương pháp: Standard Deviation Bands — Band Position = (Price − Lower) / (Upper − Lower)*")

    bands = d["val_bands"]
    if not bands.empty:
        fig_bands = go.Figure()
        fig_bands.add_trace(go.Scatter(
            x=bands["Quarter"], y=bands["Upper_Band"],
            fill=None, mode="lines", name="Cận Trên (Đắt)",
            line=dict(color="rgba(211,47,47,0.7)", dash="dot")
        ))
        fig_bands.add_trace(go.Scatter(
            x=bands["Quarter"], y=bands["Lower_Band"],
            fill="tonexty", mode="lines", name="Cận Dưới (Rẻ)",
            line=dict(color="rgba(46,125,50,0.7)", dash="dot"),
            fillcolor="rgba(200,230,201,0.3)"
        ))
        fig_bands.add_trace(go.Scatter(
            x=bands["Quarter"], y=bands["Fair_Value"],
            mode="lines", name="Fair Value (Trung bình)",
            line=dict(color="#1565C0", dash="dash", width=2)
        ))
        fig_bands.add_trace(go.Scatter(
            x=bands["Quarter"], y=bands["Price"],
            mode="lines+markers", name="Giá Thực tế",
            line=dict(color="#000000", width=3)
        ))
        fig_bands.update_layout(
            title="Giá Thực tế so với Dải Định giá Lịch sử",
            xaxis_title="Quý", yaxis_title="Giá (VNĐ)",
            legend=dict(orientation="h", y=-0.2),
            height=450,
        )
        st.plotly_chart(fig_bands, use_container_width=True, key="valuation_bands")

        # Band Position
        if "Band_Position" in bands.columns:
            bp = bands.dropna(subset=["Band_Position"])
            fig_pos = px.area(bp, x="Quarter", y="Band_Position",
                              title="Vị trí Giá trong Dải (0 = Đáy, 1 = Đỉnh)",
                              labels={"Band_Position": "Vị trí (0–1)"})
            fig_pos.add_hline(y=0.5, line_dash="dash", line_color="orange",
                               annotation_text="Trung tính")
            fig_pos.add_hline(y=0.8, line_dash="dash", line_color="red",
                               annotation_text="Vùng Đắt")
            fig_pos.add_hline(y=0.2, line_dash="dash", line_color="green",
                               annotation_text="Vùng Rẻ")
            fig_pos.update_layout(height=320)
            st.plotly_chart(fig_pos, use_container_width=True, key="band_position")
    else:
        st.info("Chưa có Valuation Bands. Hãy chạy `python main.py` để tạo.")

    # ── 3.2: DCF Sensitivity Heatmap ─────────────────────────────────────
    st.divider()
    st.subheader("3.2 Ma trận Độ nhạy DCF (Sensitivity Heatmap)")
    st.markdown("*Phương pháp: FCFF Model + Scenario Simulation — biến thiên WACC (±0.5%) & g (±0.5%)*")

    dcf_s = d["dcf_sens"]
    if not dcf_s.empty:
        dcf_s = dcf_s.set_index("WACC")
        numeric_cols = dcf_s.columns
        dcf_vals = dcf_s[numeric_cols].apply(pd.to_numeric, errors="coerce")
        fig_hm = px.imshow(
            dcf_vals,
            text_auto=".0f",
            color_continuous_scale="RdYlGn",
            title="Ma trận Độ nhạy: Equity Value theo WACC × g (tỷ VNĐ)",
            labels=dict(x="Tốc độ tăng trưởng dài hạn (g)", y="WACC", color="Equity Value"),
            aspect="auto",
        )
        fig_hm.update_layout(height=380)
        st.plotly_chart(fig_hm, use_container_width=True, key="dcf_heatmap")

        # DCF Summary
        dcf_v = d["dcf_val"]
        if not dcf_v.empty:
            row = dcf_v.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("WACC", f"{row.get('WACC_Pct', 0):.2f}%")
            c2.metric("FCFF Growth", f"{row.get('Growth_Pct', 0):.2f}%")
            c3.metric("Enterprise Value", f"{row.get('Enterprise_Value', 0):,.0f} tỷ")
            c4.metric("Equity Value", f"{row.get('Equity_Value', 0):,.0f} tỷ")
    else:
        st.info("Chưa có DCF Sensitivity. Hãy chạy `python main.py` để tạo.")

    # ── 3.3: Target Price Scenarios ───────────────────────────────────────
    st.divider()
    st.subheader("3.3 Kịch bản Giá Mục tiêu (Bear / Base / Bull)")

    tp = d["target_price"]
    if not tp.empty:
        col_chart, col_table = st.columns([2, 1])
        with col_chart:
            fig_tp = go.Figure()
            colors_tp = {"Bear": "#D32F2F", "Base": "#1565C0", "Bull": "#2E7D32"}
            for _, row in tp.iterrows():
                sc = row["Scenario"]
                fig_tp.add_trace(go.Bar(
                    name=sc, x=[sc], y=[row["Avg_Target"]],
                    marker_color=colors_tp.get(sc, "#888"),
                    text=f"{row['Avg_Target']:,.0f} VNĐ<br>({row['Upside_Pct']:+.1f}%)",
                    textposition="auto",
                ))
            fig_tp.update_layout(
                title="Target Price (Tổng hợp 3 phương pháp)",
                yaxis_title="Giá (VNĐ)", showlegend=False, height=360,
            )
            st.plotly_chart(fig_tp, use_container_width=True, key="target_price_chart")
        with col_table:
            st.markdown("**Chi tiết Kịch bản:**")
            st.dataframe(tp[["Scenario", "Avg_Target", "Upside_Pct"]],
                         hide_index=True, use_container_width=True)

        # Multi-multiple
        mult = d["multiples"]
        if not mult.empty:
            st.divider()
            st.subheader("3.4 Định giá So sánh Bội số (P/E, P/B, EV/EBITDA)")
            mult_cols = [c for c in ["Quarter", "PE_Ratio", "PB_Ratio", "EV_EBITDA", "PS_Ratio"] if c in mult.columns]
            if len(mult_cols) > 1:
                melt = mult[mult_cols].melt(id_vars="Quarter", var_name="Bội số", value_name="Giá trị")
                fig_mult = px.line(melt, x="Quarter", y="Giá trị", color="Bội số",
                                   title="Diễn biến Bội số Định giá theo Quý", markers=True)
                fig_mult.update_layout(height=380)
                st.plotly_chart(fig_mult, use_container_width=True, key="multiples_chart")
    else:
        st.info("Chưa có Target Price. Hãy chạy `python main.py` để tạo.")


# ═════════════════════════════════════════════════════════════════════════════
# CHƯƠNG 4: Phân tích Cấu trúc & Chu kỳ
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Chương 4: Phân tích Cấu trúc & Chu kỳ Chung")
    st.markdown("**Trọng tâm:** Bóc tách chu kỳ và trọng số tác động — Giải thích *\"tại sao\"* và *\"đang ở đâu trong chu kỳ\"*.")

    # ── 4.1: STL Decomposition ────────────────────────────────────────────
    st.divider()
    st.subheader("4.1 Phân rã STL (STL Decomposition Plot)")
    st.markdown("*Phương pháp: Seasonal-Trend decomposition using Loess — Y = Trend + Seasonal + Remainder*")

    decomp4 = d["cycle_decomp"]
    if not decomp4.empty:
        metrics_all = decomp4["Chỉ số"].unique().tolist()
        selected_metric = st.selectbox("Chọn chỉ số để phân rã:", metrics_all, key="stl_metric")
        sub_stl = decomp4[decomp4["Chỉ số"] == selected_metric].copy()

        fig_stl = go.Figure()
        components = [
            ("Giá trị gốc", "#555555", "dot",   "Dữ liệu gốc"),
            ("Trend",       "#1565C0", "solid",  "Trend (Xu hướng dài hạn)"),
            ("Seasonal",    "#2E7D32", "dash",   "Seasonal (Thời vụ)"),
            ("Residual",    "#D32F2F", "dashdot","Residual (Nhiễu)"),
        ]
        for col, color, dash, label in components:
            if col in sub_stl.columns:
                fig_stl.add_trace(go.Scatter(
                    x=sub_stl["Quý"], y=sub_stl[col],
                    mode="lines", name=label,
                    line=dict(color=color, dash=dash, width=2),
                ))
        fig_stl.update_layout(
            title=f"STL Decomposition — {selected_metric}",
            yaxis_title="Giá trị", xaxis_title="Quý",
            legend=dict(orientation="h", y=-0.22),
            height=440,
        )
        st.plotly_chart(fig_stl, use_container_width=True, key="stl_decomp_chart")
    else:
        st.info("Chưa có STL data. Hãy chạy `python main.py` để tạo.")

    # ── 4.2: Lead-Lag CCF Heatmap ─────────────────────────────────────────
    st.divider()
    st.subheader("4.2 Ma trận Tương quan Trễ (Lead-Lag Heatmap)")
    st.markdown("*Phương pháp: Cross-Correlation Function (CCF) — Lag −4 đến +4 quý. Lag > 0 = biến cấu trúc dẫn trước.*")

    ccf = d["cycle_ccf"]
    if not ccf.empty:
        ccf["Cặp"] = ccf["Chỉ số Hiệu quả (Trend)"] + " × " + ccf["Biến Cấu trúc"]
        pivot_ccf = ccf.pivot_table(
            index="Cặp", columns="Lag (quý)",
            values="Tương quan (CCF)", aggfunc="mean"
        ).fillna(0)
        fig_ccf = px.imshow(
            pivot_ccf,
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            zmin=-1, zmax=1,
            aspect="auto",
            title="CCF Heatmap: Tương quan Trend × Biến Cấu trúc theo Lag",
            labels=dict(x="Lag (quý)", y="Cặp phân tích", color="CCF"),
        )
        fig_ccf.update_layout(height=420)
        st.plotly_chart(fig_ccf, use_container_width=True, key="ccf_heatmap_ch4")

        # Best lag table
        best_lags = (
            ccf.dropna(subset=["Tương quan (CCF)"])
            .assign(abs_ccf=lambda df: df["Tương quan (CCF)"].abs())
            .sort_values("abs_ccf", ascending=False)
            .groupby("Cặp", as_index=False).first()
            .sort_values("abs_ccf", ascending=False)
            [["Cặp", "Lag (quý)", "Tương quan (CCF)"]]
            .rename(columns={"Tương quan (CCF)": "CCF tốt nhất"})
        )
        st.markdown("**Lag tối ưu (|CCF| lớn nhất) cho từng cặp:**")
        st.dataframe(best_lags, hide_index=True, use_container_width=True)

        cycle_md = d["cycle_md"]
        if cycle_md:
            with st.expander("📋 Báo cáo CCF đầy đủ"):
                st.markdown(cycle_md)
    else:
        st.info("Chưa có CCF data. Hãy chạy `python main.py` để tạo.")

    # ── 4.3: Factor Importance (Dual-Auditor) ─────────────────────────────
    st.divider()
    st.subheader("4.3 Trọng số Nhân tố Cấu trúc (Factor Importance)")
    st.markdown("*Mô hình: PLSR (VIP Score) + ElasticNet (Coef) — Giải thích chiều hướng & tầm quan trọng*")

    impact = d["impact"]
    metrics_df = d["metrics"]
    if not impact.empty:
        targets_list = impact["Target"].unique().tolist()
        sel_target = st.selectbox("Chọn mảng phân tích:", targets_list, key="impact_target")
        subset = impact[impact["Target"] == sel_target].copy()
        subset["Feature"] = subset["Feature"].str.strip()
        subset = subset.sort_values("PLSR_Mean_VIP", ascending=True)

        col_vip, col_coef = st.columns(2)
        with col_vip:
            fig_vip = px.bar(subset, x="PLSR_Mean_VIP", y="Feature", orientation="h",
                             title="PLSR VIP Score (Tầm quan trọng tổng thể)",
                             text_auto=".2f")
            fig_vip.add_vline(x=1.0, line_dash="dash", line_color="red",
                               annotation_text="VIP = 1 (Ngưỡng quan trọng)")
            fig_vip.update_traces(marker_color=[
                "#D32F2F" if v > 1 else "#1565C0" for v in subset["PLSR_Mean_VIP"]
            ])
            fig_vip.update_layout(height=420, yaxis_title=None)
            st.plotly_chart(fig_vip, use_container_width=True, key="vip_ch4")
        with col_coef:
            fig_coef = px.bar(subset, x="ElasticNet_Mean_Coef", y="Feature", orientation="h",
                              title="ElasticNet Hệ số (Chiều hướng +/− tác động)",
                              text_auto=".3f")
            fig_coef.add_vline(x=0, line_dash="solid", line_color="black")
            fig_coef.update_traces(marker_color=[
                "#2E7D32" if c > 0 else "#D32F2F" for c in subset["ElasticNet_Mean_Coef"]
            ])
            fig_coef.update_layout(height=420, yaxis_title=None)
            st.plotly_chart(fig_coef, use_container_width=True, key="coef_ch4")

        # OOS R² stability
        if not metrics_df.empty:
            st.markdown("**Độ ổn định Trọng số (OOS R²):** Giá trị dương = cấu trúc nhân tố ổn định qua thời gian")
            oos_melt = metrics_df.melt(
                id_vars=["Chỉ tiêu (Target)"],
                value_vars=["EN_OOS_R2", "PLS_OOS_R2"],
                var_name="Mô hình", value_name="OOS R²"
            )
            fig_oos = px.bar(oos_melt, x="Chỉ tiêu (Target)", y="OOS R²",
                             color="Mô hình", barmode="group",
                             title="OOS R² — ElasticNet vs PLSR",
                             color_discrete_map={"EN_OOS_R2": "#1565C0", "PLS_OOS_R2": "#D32F2F"},
                             text_auto=".2f")
            fig_oos.add_hline(y=0, line_dash="dash", line_color="green",
                               annotation_text="Ngưỡng R²=0")
            fig_oos.update_layout(height=360)
            st.plotly_chart(fig_oos, use_container_width=True, key="oos_ch4")
    else:
        st.info("Chưa có Factor Importance data. Hãy chạy `python main.py` để tạo.")

    # ── 4.x: Sensitivity Line Model (What-if) ────────────────────────────
    st.divider()
    st.subheader("4.x Mô hình Giả định Tác động Cấu trúc (Sensitivity Line)")
    st.markdown("*Phương pháp: What-if Analysis — Δ Target ≈ Σ (Δ Feature × ElasticNet Coef). Chỉ dùng biến VIP > 1.*")

    if not impact.empty:
        # Lọc biến VIP > 1
        vip_vars = impact[impact["PLSR_Mean_VIP"] > 1.0].copy()
        if not vip_vars.empty:
            # Phân nhóm theo Target
            target_groups = vip_vars["Target"].unique().tolist()
            sel_sens_target = st.selectbox("Chọn mảng phân tích:", target_groups, key="sens_target")
            sens_subset = vip_vars[vip_vars["Target"] == sel_sens_target].head(4)

            st.markdown("**Điều chỉnh mức thay đổi (%) cho các biến cấu trúc trọng yếu:**")
            slider_values = {}
            cols_slider = st.columns(len(sens_subset))
            for i, (_, row) in enumerate(sens_subset.iterrows()):
                feat_name = row["Feature"]
                coef = row["ElasticNet_Mean_Coef"]
                with cols_slider[i]:
                    val = st.slider(
                        f"{feat_name}",
                        min_value=-50.0, max_value=50.0, value=0.0, step=5.0,
                        key=f"sens_{i}_{feat_name[:15]}",
                        help=f"Coef: {coef:.3f} | VIP: {row['PLSR_Mean_VIP']:.2f}"
                    )
                    slider_values[feat_name] = (val, coef)

            # Tính Δ target
            delta_positive = sum(abs(c) * 20 for _, (_, c) in slider_values.items())  # +20% mỗi biến
            delta_negative = sum(-abs(c) * 20 for _, (_, c) in slider_values.items())  # -20% mỗi biến
            delta_custom = sum(v * c for _, (v, c) in slider_values.items())

            # Vẽ 3 đường
            scenarios = ["Tiêu cực\n(−20% tất cả)", "Hiện tại\n(Tùy chỉnh)", "Tích cực\n(+20% tất cả)"]
            deltas = [delta_negative, delta_custom, delta_positive]

            fig_sens = go.Figure()
            fig_sens.add_trace(go.Scatter(
                x=scenarios, y=[0 + d for d in deltas],
                mode="lines+markers+text",
                text=[f"{d:+.3f}" for d in deltas],
                textposition="top center",
                line=dict(width=3, color="#1565C0"),
                marker=dict(size=12),
                name="Δ Mục tiêu"
            ))
            fig_sens.add_hline(y=0, line_dash="dash", line_color="gray",
                               annotation_text="Baseline (không thay đổi)")
            fig_sens.update_layout(
                title=f"Sensitivity Line — {sel_sens_target}",
                yaxis_title="Δ Mục tiêu (đơn vị chuẩn hóa)",
                height=400,
                showlegend=False,
            )
            st.plotly_chart(fig_sens, use_container_width=True, key="sensitivity_line")

            # Giải thích
            if delta_custom != 0:
                direction = "tích cực" if delta_custom > 0 else "tiêu cực"
                st.info(f"📊 Với mức điều chỉnh hiện tại, mục tiêu `{sel_sens_target}` thay đổi **{delta_custom:+.4f}** ({direction}).")
            else:
                st.info("📊 Kéo các thanh trượt để mô phỏng tác động thay đổi cơ cấu.")
        else:
            st.info("Không có biến nào đạt VIP > 1 để phân tích sensitivity.")


# ═════════════════════════════════════════════════════════════════════════════
# BÁO CÁO: Luận điểm, Phương pháp luận, Kết luận
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Báo cáo Tổng hợp: Luận điểm, Phương pháp & Kết luận")

    # ── Composite Signal ──────────────────────────────────────────────────
    rec = d["rec"]
    if rec:
        st.subheader("🎯 Tín hiệu Đầu tư Tổng hợp (Composite Signal)")
        score   = rec.get("Composite_Score", 0)
        verdict = rec.get("Verdict", "N/A")

        col_s, col_v = st.columns([1, 2])
        with col_s:
            # Gauge cho composite score
            fig_score = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={"text": "Composite Score"},
                gauge={
                    "axis": {"range": [-100, 100]},
                    "bar":  {"color": "#1565C0"},
                    "steps": [
                        {"range": [-100, -15], "color": "#FFEBEE"},
                        {"range": [-15, 15],   "color": "#FFF9C4"},
                        {"range": [15, 40],    "color": "#E8F5E9"},
                        {"range": [40, 100],   "color": "#C8E6C9"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.75, "value": score}
                }
            ))
            fig_score.update_layout(height=260)
            st.plotly_chart(fig_score, use_container_width=True, key="score_gauge")

        with col_v:
            # Verdict badge
            colors_v = {"MUA MẠNH": "🟢", "MUA": "🔵", "TRUNG LẬP": "🟡", "BÁN": "🔴"}
            icon = next((v for k, v in colors_v.items() if k in verdict.upper()), "⚪")
            st.metric("Khuyến nghị", f"{icon} {verdict}", f"Score: {score}/100")
            st.info("💡 **Thang điểm:** > +40: MUA MẠNH | +15~+40: MUA | −15~+15: TRUNG LẬP | < −15: BÁN")
            comps = rec.get("Components", {})
            if comps:
                comp_df = pd.DataFrame(list(comps.items()), columns=["Thành phần", "Điểm"])
                fig_comp = px.bar(comp_df, x="Điểm", y="Thành phần", orientation="h",
                                  title="Cơ cấu Composite Score", text_auto=".1f",
                                  color="Điểm", color_continuous_scale="RdYlGn",
                                  color_continuous_midpoint=0)
                fig_comp.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_comp, use_container_width=True, key="comp_bars")
    else:
        st.warning("Chưa có Composite Signal. Hãy chạy `python main.py` để tạo.")

    # ── Phương pháp luận ──────────────────────────────────────────────────
    st.divider()
    st.subheader("📐 Phương pháp luận & Khung Phân tích")

    method_data = [
        {
            "Chương": "1 — Sức khỏe Nền tảng",
            "Câu hỏi nghiên cứu": "Dữ liệu có đáng tin cậy? Cơ cấu tài chính trông như thế nào?",
            "Mô hình hạch tâm": "Thống kê định tính (Audit Rate) + Common-size Analysis",
            "Đầu ra": "Mức độ rủi ro nền tảng",
        },
        {
            "Chương": "2 — Hiệu suất & Chu kỳ",
            "Câu hỏi nghiên cứu": "ROE tăng/giảm do đâu? Lợi nhuận có được hỗ trợ bởi tiền mặt? Chu kỳ có đồng bộ?",
            "Mô hình hạch tâm": "DuPont 3 nhân tố + Accrual Analysis (Sloan) + STL Z-score Overlay",
            "Đầu ra": "Lệch pha giữa các biến tài chính",
        },
        {
            "Chương": "3 — Định giá & Độ nhạy",
            "Câu hỏi nghiên cứu": "Cổ phiếu đang đắt hay rẻ? Biên an toàn là bao nhiêu?",
            "Mô hình hạch tâm": "DCF (FCFF) + P/S Regression + Mean Reversion Bands",
            "Đầu ra": "Khoảng Target Price Bear/Base/Bull",
        },
        {
            "Chương": "4 — Cấu trúc & Chu kỳ",
            "Câu hỏi nghiên cứu": "Nhân tố nào dẫn dắt hiệu quả? Biến cấu trúc nào báo hiệu trước?",
            "Mô hình hạch tâm": "STL Decomposition + Cross-Correlation (CCF) + PLSR VIP + ElasticNet",
            "Đầu ra": "Bản đồ nhân tố & Chỉ báo dẫn dắt (Leading Indicators)",
        },
    ]
    method_df = pd.DataFrame(method_data)
    st.dataframe(method_df, hide_index=True, use_container_width=True)

    st.markdown("""
    > **Ghi chú triết lý:** Hệ thống sử dụng PLSR VIP và ElasticNet **để giải thích cơ cấu**, không để dự báo điểm.
    > OOS R² đo **độ ổn định của trọng số tác động** qua thời gian, không phải sức mạnh dự báo.
    > Module kiểm tra Overfitting (R² IS/OOS truyền thống) đã được loại bỏ.
    """)

    # ── Kết luận tổng hợp từ report.md ───────────────────────────────────
    st.divider()
    st.subheader("📄 Báo cáo Chi tiết (Dual-Auditor)")
    report_md = d["report_md"]
    if report_md:
        with st.expander("Xem Báo cáo Dual-Auditor đầy đủ", expanded=False):
            st.markdown(report_md)
    else:
        st.info("Chưa có analysis_report.md.")

    st.subheader("📄 Báo cáo Phân tích Chu kỳ (STL + CCF)")
    cycle_md5 = d["cycle_md"]
    if cycle_md5:
        with st.expander("Xem Báo cáo Chu kỳ đầy đủ", expanded=False):
            st.markdown(cycle_md5)
    else:
        st.info("Chưa có cycle_report.md.")
