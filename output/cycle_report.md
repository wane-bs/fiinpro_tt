# Báo cáo Phân tích Chu kỳ Cấu trúc (STL + Cross-Correlation)
> Phân tích dựa trên 64 quý dữ liệu. STL period=4 (chu kỳ quý). Lag dương = biến cấu trúc **dẫn trước** chỉ số hiệu quả.

## Bản đồ Tác động: Lag có Tương quan Cao nhất

| Chỉ số Hiệu quả (Trend) | Biến Cấu trúc | Lag (quý) | Tương quan | Diễn giải |
|:---|:---|:---:|:---:|:---|
| Biên LN Gộp (%) | HTK/TTS (%) | +1 | 🔴 -0.961 | `HTK/TTS (%)` dẫn trước `Biên LN Gộp (%)` 1 quý (tương quan nghịch) |
| Biên LN Gộp (%) | Phải thu/TTS (%) | +4 | 🔴 -0.901 | `Phải thu/TTS (%)` dẫn trước `Biên LN Gộp (%)` 4 quý (tương quan nghịch) |
| ROE (%) | HTK/TTS (%) | +4 | 🔴 0.516 | `HTK/TTS (%)` dẫn trước `ROE (%)` 4 quý (tương quan thuận) |
| ROE (%) | Phải thu/TTS (%) | +4 | 🟡 0.453 | `Phải thu/TTS (%)` dẫn trước `ROE (%)` 4 quý (tương quan thuận) |
| Biên LN Gộp (%) | Nợ NH/TTS (%) | +4 | 🟡 0.436 | `Nợ NH/TTS (%)` dẫn trước `Biên LN Gộp (%)` 4 quý (tương quan thuận) |
| Biên LN Gộp (%) | Nợ/TTS (%) | +0 | 🟡 -0.334 | `Nợ/TTS (%)` và `Biên LN Gộp (%)` biến động đồng thời (lag=0) |
| ROA (%) | HTK/TTS (%) | +4 | 🟡 0.314 | `HTK/TTS (%)` dẫn trước `ROA (%)` 4 quý (tương quan thuận) |
| ROA (%) | Nợ/TTS (%) | +1 | ⚪ -0.290 | `Nợ/TTS (%)` dẫn trước `ROA (%)` 1 quý (tương quan nghịch) |
| ROE (%) | Nợ/TTS (%) | +0 | ⚪ 0.241 | `Nợ/TTS (%)` và `ROE (%)` biến động đồng thời (lag=0) |
| ROA (%) | Phải thu/TTS (%) | +4 | ⚪ 0.237 | `Phải thu/TTS (%)` dẫn trước `ROA (%)` 4 quý (tương quan thuận) |
| ROE (%) | Nợ NH/TTS (%) | +4 | ⚪ -0.196 | `Nợ NH/TTS (%)` dẫn trước `ROE (%)` 4 quý (tương quan nghịch) |
| ROA (%) | Nợ NH/TTS (%) | +2 | ⚪ -0.173 | `Nợ NH/TTS (%)` dẫn trước `ROA (%)` 2 quý (tương quan nghịch) |

## Giải thích Ký hiệu
- **Lag > 0:** Biến cấu trúc thay đổi *trước*, sau đó chỉ số hiệu quả phản ứng.
- **Lag < 0:** Chỉ số hiệu quả dẫn trước biến cấu trúc.
- 🔴 |CCF| > 0.5 (tương quan mạnh) | 🟡 |CCF| > 0.3 (trung bình) | ⚪ yếu

## Dữ liệu gốc
- **STL Decomposition:** `output/cycle_decomposition.csv`
- **Ma trận CCF đầy đủ:** `output/cycle_cross_correlation.csv`
