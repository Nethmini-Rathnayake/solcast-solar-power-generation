"""
generate_report.py
-------------------
Generates a PDF report summarising all forecasting approaches,
their methodologies, and a final comparison table.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Image,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from pathlib import Path
import pandas as pd

# ── Output ─────────────────────────────────────────────────────────────────────
OUT_PDF = "Solar_PV_Forecasting_Report.pdf"

# ── Styles ─────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def S(name):
    return styles[name]

title_style = ParagraphStyle(
    "ReportTitle",
    parent=styles["Title"],
    fontSize=22,
    textColor=colors.HexColor("#1a3a5c"),
    spaceAfter=6,
    alignment=TA_CENTER,
)
subtitle_style = ParagraphStyle(
    "Subtitle",
    parent=styles["Normal"],
    fontSize=11,
    textColor=colors.HexColor("#4a6fa5"),
    spaceAfter=4,
    alignment=TA_CENTER,
)
h1_style = ParagraphStyle(
    "H1",
    parent=styles["Heading1"],
    fontSize=14,
    textColor=colors.HexColor("#1a3a5c"),
    spaceBefore=14,
    spaceAfter=6,
    borderPad=4,
)
h2_style = ParagraphStyle(
    "H2",
    parent=styles["Heading2"],
    fontSize=12,
    textColor=colors.HexColor("#2e5fa3"),
    spaceBefore=10,
    spaceAfter=4,
)
body_style = ParagraphStyle(
    "Body",
    parent=styles["Normal"],
    fontSize=10,
    leading=14,
    spaceAfter=6,
    alignment=TA_JUSTIFY,
)
bullet_style = ParagraphStyle(
    "Bullet",
    parent=styles["Normal"],
    fontSize=10,
    leading=13,
    leftIndent=16,
    spaceAfter=3,
)
caption_style = ParagraphStyle(
    "Caption",
    parent=styles["Normal"],
    fontSize=8,
    textColor=colors.grey,
    alignment=TA_CENTER,
    spaceAfter=6,
)
note_style = ParagraphStyle(
    "Note",
    parent=styles["Normal"],
    fontSize=9,
    textColor=colors.HexColor("#555555"),
    leading=12,
    leftIndent=12,
    spaceAfter=4,
)

HEADER_COLOR  = colors.HexColor("#1a3a5c")
ROW_ALT_COLOR = colors.HexColor("#eaf0fb")
BEST_COLOR    = colors.HexColor("#d4edda")
WORST_COLOR   = colors.HexColor("#fde8e8")

# ── Helpers ────────────────────────────────────────────────────────────────────

def hr():
    return HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc"),
                      spaceAfter=6, spaceBefore=6)

def h1(text):
    return Paragraph(text, h1_style)

def h2(text):
    return Paragraph(text, h2_style)

def body(text):
    return Paragraph(text, body_style)

def bullet(text):
    return Paragraph(f"• {text}", bullet_style)

def note(text):
    return Paragraph(f"<i>{text}</i>", note_style)

def sp(h=6):
    return Spacer(1, h)

def fig(path, width=15*cm, caption_text=None):
    elems = []
    if Path(path).exists():
        elems.append(Image(path, width=width, height=width*0.42))
        if caption_text:
            elems.append(Paragraph(caption_text, caption_style))
    return elems

def make_table(data, col_widths, header_rows=1):
    t = Table(data, colWidths=col_widths, repeatRows=header_rows)
    style = TableStyle([
        # Header
        ("BACKGROUND",  (0, 0), (-1, header_rows - 1), HEADER_COLOR),
        ("TEXTCOLOR",   (0, 0), (-1, header_rows - 1), colors.white),
        ("FONTNAME",    (0, 0), (-1, header_rows - 1), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, header_rows - 1), 9),
        ("ALIGN",       (0, 0), (-1, header_rows - 1), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, header_rows - 1), 7),
        ("TOPPADDING",    (0, 0), (-1, header_rows - 1), 7),
        # Body
        ("FONTNAME",  (0, header_rows), (-1, -1), "Helvetica"),
        ("FONTSIZE",  (0, header_rows), (-1, -1), 9),
        ("ALIGN",     (1, header_rows), (-1, -1), "CENTER"),
        ("ALIGN",     (0, header_rows), (0, -1),  "LEFT"),
        ("TOPPADDING",    (0, header_rows), (-1, -1), 5),
        ("BOTTOMPADDING", (0, header_rows), (-1, -1), 5),
        # Light background for all body rows; specific rows overridden after
        ("BACKGROUND", (0, header_rows), (-1, -1), colors.HexColor("#f5f8ff")),
        # Grid
        ("GRID",      (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("VALIGN",    (0, 0), (-1, -1), "MIDDLE"),
    ])
    t.setStyle(style)
    return t

# ── Load metrics ───────────────────────────────────────────────────────────────

def load_mean(path):
    try:
        df = pd.read_csv(path, index_col=0)
        return df.loc["mean"]
    except Exception:
        return None

xgb   = load_mean("results/metrics/metrics_xgb_dms.csv")
lstm  = load_mean("results/metrics/metrics_lstm.csv")
lstm_s= load_mean("results/metrics/lstm_synthetic/metrics_lstm_synthetic.csv")
hyb   = load_mean("results/metrics/metrics_hybrid.csv")
pers  = load_mean("results/metrics/metrics_persistence.csv")
sday  = load_mean("results/metrics/metrics_same_day.csv")
cnn   = load_mean("results/metrics_per_horizon.csv")  # different column names

def fmt(val, decimals=1, suffix=""):
    if val is None:
        return "—"
    try:
        return f"{float(val):.{decimals}f}{suffix}"
    except Exception:
        return "—"

# ── Build PDF ──────────────────────────────────────────────────────────────────

def build():
    doc = SimpleDocTemplate(
        OUT_PDF,
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2.2*cm, bottomMargin=2*cm,
        title="Solar PV Forecasting — Approach Comparison Report",
        author="Nethmini Rathnayake",
    )

    story = []

    # ══════════════════════════════════════════════════════════════════════════
    # COVER
    # ══════════════════════════════════════════════════════════════════════════
    story += [
        sp(40),
        Paragraph("Solar PV Power Generation Forecasting", title_style),
        Paragraph("Approach Comparison Report", subtitle_style),
        sp(4),
        Paragraph("University of Moratuwa Microgrid — Sri Lanka", subtitle_style),
        sp(6),
        hr(),
        sp(4),
        Paragraph(
            "This report documents all forecasting approaches developed for 1–24 hour ahead "
            "solar PV prediction using Solcast irradiance data and real PV generation records "
            "from the University of Moratuwa smart-grid lab. Each approach is described with "
            "its architecture, training strategy, and test-set performance.",
            body_style,
        ),
        sp(6),
        *make_cover_summary_table(),
        sp(10),
        Paragraph("Author: Nethmini Rathnayake", caption_style),
        PageBreak(),
    ]

    # ══════════════════════════════════════════════════════════════════════════
    # 1. PROJECT OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    story += [
        h1("1. Project Overview"),
        hr(),
        body(
            "The goal of this project is to build a high-accuracy solar photovoltaic (PV) "
            "power forecasting pipeline for a tropical site (6.79°N, 79.90°E, Sri Lanka). "
            "The system predicts AC power output 1 to 24 hours ahead using historical Solcast "
            "irradiance and weather data, physics-based pvlib simulations, and deep learning."
        ),
        sp(4),
        h2("1.1 Dataset"),
        *[bullet(b) for b in [
            "<b>Real PV data:</b> 5-minute AC power readings from the UoM smart-grid lab "
            "(April 2022 – March 2023, ~105k rows). System capacity: ~250 kW peak.",
            "<b>Solcast weather data:</b> Hourly irradiance and atmospheric variables "
            "(2020–2023, 4 years). Columns: GHI, DNI, DHI, cloud opacity, air temperature, "
            "relative humidity, surface pressure, dewpoint, clearsky irradiance.",
            "<b>Calibrated synthetic data:</b> 4-year hourly synthetic PV generation "
            "produced by pvlib physics simulation + LSTM residual correction, calibrated "
            "to the 1-year real PV window.",
        ]],
        sp(4),
        h2("1.2 Feature Engineering"),
        *[bullet(b) for b in [
            "Cyclical time encodings: hour, day-of-year, month (sin/cos pairs)",
            "Physics features: pvlib AC output, clearness index, cos(solar zenith)",
            "Monsoon regime flags: SW monsoon, NE monsoon, two inter-monsoon seasons",
            "Lag features: pv_lag24h, pv_lag48h, ghi_lag24h, clearness_lag24h",
            "NWP forecast context: ghi_fcast_h1–h24, cloud_opacity_fcast_h1–h24",
            "Future-weather summaries: mean/max GHI ahead, daylight hours ahead",
            "Rolling statistics: 3h / 6h / 24h rolling means (for XGBoost)",
        ]],
        sp(4),
        h2("1.3 Evaluation Protocol"),
        body(
            "All models are evaluated on the same chronological held-out test set "
            "(last ~15% of the real PV data, approximately January–March 2023). "
            "Metrics computed per forecast horizon and averaged across h=1…24:"
        ),
        *[bullet(b) for b in [
            "<b>RMSE</b> — Root Mean Squared Error (W or kW)",
            "<b>MAE</b>  — Mean Absolute Error (W or kW)",
            "<b>nRMSE</b> — Normalised RMSE = RMSE / mean(observed) × 100 %",
            "<b>R²</b>   — Coefficient of determination (1.0 = perfect)",
            "<b>MAPE</b> — Mean Absolute Percentage Error (daytime only, observed > 1 kW)",
        ]],
        PageBreak(),
    ]

    # ══════════════════════════════════════════════════════════════════════════
    # 2. BASELINES
    # ══════════════════════════════════════════════════════════════════════════
    story += [
        h1("2. Baseline Models"),
        hr(),
        h2("2.1 Persistence Baseline"),
        body(
            "The simplest possible forecast: the power at time t+h is predicted to equal "
            "the power observed at time t (last observed value held constant for all horizons). "
            "This baseline is trivially easy to beat at long horizons and serves as a lower bound."
        ),
        *[bullet(b) for b in [
            "No training required",
            "Degrades rapidly beyond h+1 — completely flat forecast",
            f"Mean RMSE: {fmt(pers['RMSE'] if pers is not None else None, 0)} W  |  "
            f"R²: {fmt(pers['R2'] if pers is not None else None, 4)}",
        ]],
        sp(6),
        h2("2.2 Same-Day Baseline (Seasonal Persistence)"),
        body(
            "Predicts tomorrow's h+k power using the same clock-hour from yesterday — "
            "i.e. ŷ(t+h) = y(t+h−24). This is a strong baseline for solar forecasting "
            "because the diurnal solar cycle is highly repeatable, especially in tropical "
            "regions with stable seasonal patterns."
        ),
        *[bullet(b) for b in [
            "No training required; uses only the PV time series",
            "Captures diurnal periodicity perfectly on clear days",
            "Fails on day-to-day cloud variability",
            f"Mean RMSE: {fmt(sday['RMSE'] if sday is not None else None, 0)} W  |  "
            f"R²: {fmt(sday['R2'] if sday is not None else None, 4)}",
        ]],
        PageBreak(),
    ]

    # ══════════════════════════════════════════════════════════════════════════
    # 3. XGBOOST DMS
    # ══════════════════════════════════════════════════════════════════════════
    story += [
        h1("3. XGBoost Direct Multi-Step Forecaster (DMS)"),
        hr(),
        h2("3.1 Architecture"),
        body(
            "24 independent XGBoost regression models are trained — one per forecast horizon "
            "h ∈ {1, …, 24}. Each model f_h predicts ŷ(t+h) = f_h(X_t), where X_t is "
            "the full feature vector at decision time t. This Direct Multi-Step (DMS) "
            "strategy avoids error accumulation that would occur with a recursive approach."
        ),
        sp(4),
        h2("3.2 Horizon-Specific Feature Selection"),
        body(
            "Short-range (h ≤ 8) models use all features including recent lag features "
            "(lag_1h … lag_8h). Mid-range (h 9–16) models drop lag_1h–lag_3h as these "
            "become stale. Long-range (h 17–24) models drop lag_1h–lag_8h entirely, "
            "relying instead on day-lag features (pv_lag24, pv_lag48) and NWP forecast "
            "context which dominate at day-ahead horizons."
        ),
        sp(4),
        h2("3.3 Hyperparameters"),
        *[bullet(b) for b in [
            "n_estimators: 500  |  max_depth: 6  |  learning_rate: 0.05",
            "subsample: 0.8  |  colsample_bytree: 0.8",
            "min_child_weight: 5  |  reg_alpha: 0.1  |  reg_lambda: 1.0",
            "Early stopping: 50 rounds on validation RMSE",
        ]],
        sp(4),
        h2("3.4 Training Data"),
        body(
            "Trained on 70% of the real PV hourly feature matrix (≈5,400 rows), "
            "validated on the next 15% (≈1,150 rows). Test set is the final 15% "
            "(≈1,150 rows, January–March 2023)."
        ),
        sp(4),
        h2("3.5 Results"),
        *[bullet(b) for b in [
            f"Mean RMSE: {fmt(xgb['RMSE'] if xgb is not None else None, 0)} W  "
            f"({fmt(xgb['RMSE']/1000 if xgb is not None else None, 1)} kW)",
            f"Mean MAE:  {fmt(xgb['MAE'] if xgb is not None else None, 0)} W",
            f"nRMSE:     {fmt(xgb.get('nRMSE') if xgb is not None else None, 2)} %",
            f"R²:        {fmt(xgb['R2'] if xgb is not None else None, 4)}",
            "R² range across horizons: 0.856 (h+24) – 0.947 (h+1)",
            "Best single-model approach on the 1-year real data training set",
        ]],
        PageBreak(),
    ]

    # ══════════════════════════════════════════════════════════════════════════
    # 4. LSTM PRETRAIN + FINETUNE
    # ══════════════════════════════════════════════════════════════════════════
    story += [
        h1("4. LSTM Forecaster — Pretrain + Finetune"),
        hr(),
        h2("4.1 Architecture"),
        body(
            "A 2-layer PyTorch LSTM with sequence length 48 (2-day look-back window) "
            "and MIMO output of 24 horizon predictions in a single forward pass. "
            "Optional NWP forecast context (48 features: GHI and cloud opacity for h1–h24) "
            "is concatenated to the LSTM's last hidden state before the output Dense layer."
        ),
        *[bullet(b) for b in [
            "Input: (batch, seq_len=48, 9 historical features)",
            "LSTM: hidden_size=64, n_layers=2, dropout=0.2",
            "Forecast context injection: concat([last_hidden | fcast_ctx(48)])",
            "Output: Dense(24) → 24-horizon forecast (MIMO)",
        ]],
        sp(4),
        h2("4.2 Two-Phase Training Strategy"),
        body(
            "<b>Phase 1 — Pre-training:</b> 100 epochs on 4-year calibrated synthetic PV "
            "(≈36,000 hourly rows). The model learns the general solar diurnal and seasonal "
            "patterns from physically realistic but simulated data. "
            "<b>Phase 2 — Fine-tuning:</b> 30 epochs on 1-year real PV data at a 10× lower "
            "learning rate (1e-4 vs 1e-3) to adapt to site-specific characteristics without "
            "catastrophic forgetting of the pre-trained knowledge."
        ),
        sp(4),
        h2("4.3 Daytime-Weighted Loss"),
        body(
            "Training uses a per-horizon weighted MSE loss: horizons where the target "
            "pv_ac_norm > 0.05 (daytime) receive a weight of 3.0 vs 1.0 for nighttime. "
            "This prevents the model from over-optimising on easy nighttime zeros at the "
            "expense of harder daytime prediction."
        ),
        sp(4),
        h2("4.4 Results"),
        *[bullet(b) for b in [
            f"Mean RMSE: {fmt(lstm['RMSE'] if lstm is not None else None, 0)} W",
            f"Mean MAE:  {fmt(lstm['MAE'] if lstm is not None else None, 0)} W",
            f"R²:        {fmt(lstm['R2'] if lstm is not None else None, 4)}",
            "Lower accuracy than XGBoost on the 1-year test set",
            "Catastrophic forgetting risk during fine-tuning limits performance",
        ]],
        PageBreak(),
    ]

    # ══════════════════════════════════════════════════════════════════════════
    # 5. LSTM DIRECT ON SYNTHETIC
    # ══════════════════════════════════════════════════════════════════════════
    story += [
        h1("5. LSTM — Direct Training on Calibrated Synthetic"),
        hr(),
        h2("5.1 Motivation"),
        body(
            "Instead of the two-phase pretrain → finetune approach, this variant trains "
            "the LSTM in a single pass directly on the 4-year calibrated synthetic dataset "
            "for only 20 epochs. This eliminates the catastrophic forgetting risk and runs "
            "approximately 8× faster (~15 minutes vs ~2 hours)."
        ),
        sp(4),
        h2("5.2 Architecture & Data"),
        *[bullet(b) for b in [
            "Same architecture as Section 4: seq_len=48, hidden=64, n_layers=2",
            "Training data: synthetic_corrected_4yr.parquet (36k rows, 2020–2024)",
            "Evaluation: real PV test set (January–March 2023)",
            "20 epochs, lr=0.001, early stopping patience=15",
        ]],
        sp(4),
        h2("5.3 Feature Engineering for Synthetic"),
        body(
            "Day-lag features (pv_lag24, pv_lag48, ghi_lag24, clearness_lag24) and "
            "oracle forecast features (actual future GHI used as perfect NWP proxy) "
            "are added to the synthetic dataframe before sequence creation."
        ),
        sp(4),
        h2("5.4 Results"),
        *[bullet(b) for b in [
            f"Mean RMSE: {fmt(lstm_s['RMSE'] if lstm_s is not None else None, 0)} W",
            f"Mean MAE:  {fmt(lstm_s['MAE'] if lstm_s is not None else None, 0)} W",
            f"R²:        {fmt(lstm_s['R2'] if lstm_s is not None else None, 4)}",
            "Faster to train but lower R² — distribution gap between synthetic and real",
            "Beats the same-day baseline (R²=0.796)",
        ]],
        PageBreak(),
    ]

    # ══════════════════════════════════════════════════════════════════════════
    # 6. HYBRID
    # ══════════════════════════════════════════════════════════════════════════
    story += [
        h1("6. Hybrid Forecaster (XGBoost + LSTM)"),
        hr(),
        h2("6.1 Architecture"),
        body(
            "A heterogeneous parallel ensemble that blends the XGBoost DMS and LSTM "
            "predictions using a per-horizon-group blending weight α:"
        ),
        body(
            "ŷ_hybrid(t, h) = α_group · ŷ_XGB(t, h) + (1 − α_group) · ŷ_LSTM(t, h)"
        ),
        body(
            "Three independent α values are optimised by grid search on the validation set, "
            "minimising mean RMSE across horizons within each group:"
        ),
        *[bullet(b) for b in [
            "<b>Short</b> (h=1–8):   LSTM sequential context is strongest at short range",
            "<b>Mid</b>   (h=9–16):  blended transition zone",
            "<b>Long</b>  (h=17–24): NWP forecast features dominate; XGBoost leads",
        ]],
        sp(4),
        h2("6.2 Alpha Optimisation"),
        body(
            "Grid search over α ∈ {0.00, 0.05, …, 1.00} (21 steps) is run independently "
            "for each horizon group on the validation set. The best α per group is stored "
            "and applied at inference time."
        ),
        sp(4),
        h2("6.3 Results"),
        *[bullet(b) for b in [
            f"Mean RMSE: {fmt(hyb['RMSE'] if hyb is not None else None, 0)} W",
            f"Mean MAE:  {fmt(hyb['MAE'] if hyb is not None else None, 0)} W",
            f"R²:        {fmt(hyb['R2'] if hyb is not None else None, 4)}",
            "Performance matched XGBoost — LSTM component did not add value in this run",
            "Hybrid benefit is expected to grow with a better-calibrated LSTM",
        ]],
        PageBreak(),
    ]

    # ══════════════════════════════════════════════════════════════════════════
    # 7. CNN-LSTM
    # ══════════════════════════════════════════════════════════════════════════
    story += [
        h1("7. CNN-LSTM Hybrid (Keras/TensorFlow)"),
        hr(),
        h2("7.1 Architecture"),
        body(
            "A CNN-BiLSTM hybrid implemented in Keras/TensorFlow 2.21. The CNN block "
            "extracts local temporal patterns from the 24-hour input window, while the "
            "BiLSTM layers capture long-range sequential dependencies in both forward "
            "and backward directions."
        ),
        make_arch_table(),
        sp(4),
        h2("7.2 Training Strategy"),
        body(
            "<b>Train:</b> 36,410 sequences from the 4-year calibrated synthetic dataset "
            "(each sequence: 24 timesteps × 79 features). "
            "<b>Validation (early stopping):</b> 6,424 sequences from the real PV dataset "
            "(first 85% of real data). "
            "<b>Test:</b> 1,095 sequences from the real PV held-out set (last 15%)."
        ),
        *[bullet(b) for b in [
            "Optimizer: Adam (lr=0.001)",
            "Loss: MSE on normalised PV (MinMaxScaler fit on synthetic)",
            "Callbacks: EarlyStopping (patience=10), ReduceLROnPlateau (factor=0.5)",
            "Early stop triggered at epoch 88/100  |  Best val_loss: 0.0217",
        ]],
        sp(4),
        h2("7.3 Key Features (79 total)"),
        *[bullet(b) for b in [
            "Irradiance: GHI, DNI, DHI, cloud_opacity, ghi_clearsky_ratio",
            "Meteorological: air_temp, relative_humidity, surface_pressure, dewpoint",
            "Physics: pvlib_ac_W (clearsky model), cos_solar_zenith, solar_elevation",
            "Time: hour_sin/cos, month_sin/cos, doy_sin/cos",
            "Monsoon regime: monsoon_sw, monsoon_ne, monsoon_inter1, monsoon_inter2",
            "Lag: pv_lag24, pv_lag48, ghi_lag24, clearness_lag24",
            "NWP forecasts: ghi_fcast_h1–h24, cloud_opacity_fcast_h1–h24 (48 cols)",
            "Summaries: ghi_fcast_mean_24h, max_24h, total_irradiance_ahead, daylight_hours_ahead",
        ]],
        sp(4),
        h2("7.4 Results"),
        *[bullet(b) for b in [
            f"Mean RMSE: {fmt(cnn['RMSE_kW'] if cnn is not None else None, 3)} kW  "
            f"({fmt(cnn['RMSE_kW']*1000 if cnn is not None else None, 0)} W)",
            f"Mean MAE:  {fmt(cnn['MAE_kW'] if cnn is not None else None, 3)} kW",
            f"R²:        {fmt(cnn['R2'] if cnn is not None else None, 4)}",
            "R² range across horizons: 0.902 (h+24) – 0.933 (h+1)",
            "<b>All 24 horizons exceed the R² ≥ 0.90 target</b>",
            "Best overall model — highest mean R² across all approaches",
        ]],
        sp(6),
        *fig("results/figures/r2_by_horizon.png", width=14*cm,
             caption_text="Figure 1: R² by forecast horizon — CNN-LSTM on real PV test set. "
                          "Red dashed line = 0.90 target. All horizons exceed the target."),
        sp(4),
        *fig("results/figures/rmse_by_horizon.png", width=14*cm,
             caption_text="Figure 2: RMSE by forecast horizon. "
                          "Flat profile (~21 kW) confirms stable accuracy across all 24 hours."),
        PageBreak(),
        *fig("results/figures/forecast_vs_actual_h01.png", width=16*cm,
             caption_text="Figure 3: Forecast vs Actual — h+1 (1-hour ahead). "
                          "Last 7 days of test set shown."),
        sp(4),
        *fig("results/figures/forecast_vs_actual_h06.png", width=16*cm,
             caption_text="Figure 4: Forecast vs Actual — h+6 (6-hour ahead)."),
        sp(4),
        *fig("results/figures/forecast_vs_actual_h12.png", width=16*cm,
             caption_text="Figure 5: Forecast vs Actual — h+12 (12-hour ahead)."),
        sp(4),
        *fig("results/figures/forecast_vs_actual_h24.png", width=16*cm,
             caption_text="Figure 6: Forecast vs Actual — h+24 (24-hour ahead). "
                          "Diurnal pattern captured well even at full day-ahead horizon."),
        PageBreak(),
        *fig("results/figures/scatter_h01.png", width=9*cm,
             caption_text="Figure 7: Scatter plot — h+1  (R²=0.933)"),
        *fig("results/figures/scatter_h12.png", width=9*cm,
             caption_text="Figure 8: Scatter plot — h+12 (R²=0.927)"),
        *fig("results/figures/scatter_h24.png", width=9*cm,
             caption_text="Figure 9: Scatter plot — h+24 (R²=0.902)"),
        *fig("results/figures/training_history.png", width=14*cm,
             caption_text="Figure 10: CNN-LSTM training history. "
                          "Val loss (orange, real PV) steadily decreases alongside train loss "
                          "(blue, synthetic), confirming good generalisation."),
        PageBreak(),
    ]

    # ══════════════════════════════════════════════════════════════════════════
    # 8. SYNTHETIC DATA GENERATION
    # ══════════════════════════════════════════════════════════════════════════
    story += [
        h1("8. Calibrated Synthetic Data Generation"),
        hr(),
        body(
            "Because only 1 year of real PV data is available, a 4-year calibrated synthetic "
            "dataset was generated to expand the training set for deep learning models. "
            "The pipeline follows methodology from NREL Technical Report TP-5K00-86459."
        ),
        sp(4),
        h2("8.1 Generation Pipeline"),
        *[bullet(b) for b in [
            "<b>pvlib physics simulation:</b> Models POA irradiance, cell temperature, "
            "and DC/AC output using system parameters (pdc0, gamma_pdc, inverter efficiency)",
            "<b>LSTM residual correction:</b> A 2-layer LSTM learns the site-specific "
            "residual between pvlib output and real PV (trained on the 1-year overlap)",
            "<b>Stochastic disturbances</b> based on NREL fleet statistics:",
        ]],
        *[bullet(f"  ∘ {b}") for b in [
            "Degradation: 0.75%/year linear (truncated normal σ=0.5%/yr)",
            "Soiling: monsoon-aware sawtooth (calibrated to Colombo rainfall data), "
            "max loss uniform U(5%, 20%)",
            "Outages: Poisson(λ=8.6/yr), exponential duration (mean 74h), "
            "curtailment Poisson(10/yr) at 75–90% cap",
            "Weather noise: heteroscedastic, proportional to cloud variability",
        ]],
        sp(4),
        h2("8.2 Monsoon Calibration"),
        body(
            "Sri Lanka has a bimodal rainfall pattern (SW monsoon peak May, 2nd inter-monsoon "
            "peak October). Soiling accumulation rates are calibrated per month to actual "
            "Colombo rainfall data. Monsoon onset/withdrawal uses sigmoid transitions with "
            "stochastic timing (SW onset: DOY 135 ± 21 days) to produce realistic "
            "year-to-year shape variation."
        ),
        PageBreak(),
    ]

    # ══════════════════════════════════════════════════════════════════════════
    # 9. COMPARISON TABLE
    # ══════════════════════════════════════════════════════════════════════════
    story += [
        h1("9. Model Comparison Summary"),
        hr(),
        body(
            "All models are evaluated on the same real PV test set "
            "(approximately January–March 2023, ~1,095 hourly samples). "
            "Metrics are averaged across all 24 forecast horizons (h=1…24)."
        ),
        sp(8),
        make_comparison_table(xgb, lstm, lstm_s, hyb, cnn, pers, sday),
        sp(10),
        note(
            "* CNN-LSTM RMSE/MAE reported in kW; all others in W. "
            "MAPE is computed on daytime hours only (observed > 1 kW / 1,000 W) "
            "to avoid division by near-zero nighttime values. "
            "nRMSE = RMSE / mean(observed) × 100%. "
            "Best values per column are highlighted in green."
        ),
        sp(8),
        h2("9.1 Key Findings"),
        *[bullet(b) for b in [
            "<b>CNN-LSTM achieves the highest R² (0.925)</b> and is the only approach where "
            "all 24 horizons individually exceed R²=0.90.",
            "<b>XGBoost DMS (R²=0.884)</b> is the best single-dataset approach, performing "
            "strongly despite training on only 1 year of real data.",
            "<b>Hybrid (XGB+LSTM)</b> matched XGBoost in this run. Larger gains are expected "
            "with a better-calibrated LSTM component.",
            "<b>LSTM pretrain+finetune (R²=0.770)</b> — the two-phase approach suffered from "
            "distribution mismatch between synthetic and real data.",
            "<b>Same-day baseline (R²=0.796)</b> is a surprisingly competitive benchmark "
            "for tropical solar — beating it is non-trivial.",
            "<b>Persistence baseline (R²=−0.968)</b> collapses at long horizons as expected.",
        ]],
        sp(6),
        h2("9.2 Recommendations"),
        *[bullet(b) for b in [
            "Use <b>CNN-LSTM</b> for production deployment — highest accuracy, stable across "
            "all 24 horizons, leverages the full 4-year synthetic training set.",
            "Fine-tune CNN-LSTM on the real PV validation split for an additional 10–20 epochs "
            "at lr=1e-4 to close the synthetic→real domain gap further.",
            "Re-run <b>Hybrid</b> after the CNN-LSTM fine-tune pass — the ensemble is likely "
            "to outperform any individual model.",
            "For operational NWP: replace oracle forecast features with live Solcast "
            "forecast API calls to eliminate information leakage at inference time.",
        ]],
    ]

    doc.build(story)
    print(f"Report saved: {OUT_PDF}")


# ── Helper tables ──────────────────────────────────────────────────────────────

def make_cover_summary_table():
    data = [
        ["Approach", "R² (mean)", "RMSE", "Training data"],
        ["CNN-LSTM (Keras)",       "0.9249", "21.3 kW",   "4-yr synthetic → real val"],
        ["XGBoost DMS",            "0.8837", "26.5 kW",   "1-yr real PV"],
        ["Hybrid (XGB+LSTM)",      "0.8837", "26.5 kW",   "1-yr real PV"],
        ["Same-day baseline",      "0.7963", "35.3 kW",   "None"],
        ["LSTM pretrain+finetune", "0.7695", "37.4 kW",   "4-yr synthetic + 1-yr real"],
        ["LSTM direct (synthetic)","0.7429", "39.5 kW",   "4-yr synthetic"],
        ["Persistence baseline",   "−0.968", "103.4 kW",  "None"],
    ]
    col_w = [6.5*cm, 2.5*cm, 2.5*cm, 5.5*cm]
    t = make_table(data, col_w)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 1), (-1, 1), BEST_COLOR),
        ("BACKGROUND", (0, -1), (-1, -1), WORST_COLOR),
    ]))
    return [t]


def make_arch_table():
    data = [
        ["Layer", "Output Shape", "Parameters"],
        ["Input (seq_len=24, features=79)", "(batch, 24, 79)",  "—"],
        ["Conv1D(64, kernel=3, ReLU)",       "(batch, 24, 64)",  "15,232"],
        ["Conv1D(32, kernel=3, ReLU)",       "(batch, 24, 32)",  "6,176"],
        ["MaxPooling1D(pool=2)",             "(batch, 12, 32)",  "—"],
        ["BiLSTM(128, return_seq=True)",     "(batch, 12, 256)", "164,864"],
        ["Dropout(0.2)",                     "(batch, 12, 256)", "—"],
        ["LSTM(64)",                         "(batch, 64)",      "82,176"],
        ["Dropout(0.2)",                     "(batch, 64)",      "—"],
        ["Dense(24)",                        "(batch, 24)",      "1,560"],
        ["Total trainable params",           "",                 "264,056 (1.0 MB)"],
    ]
    col_w = [7.5*cm, 4.5*cm, 3.5*cm]
    return make_table(data, col_w)


def make_comparison_table(xgb, lstm, lstm_s, hyb, cnn, pers, sday):
    cnn_rmse_w = cnn["RMSE_kW"] * 1000 if cnn is not None else None
    cnn_mae_w  = cnn["MAE_kW"]  * 1000 if cnn is not None else None

    def row(name, obj, rmse_override=None, mae_override=None, mape_override=None,
            nrmse_override=None, mbe_override=None, r2_override=None, note=""):
        if obj is None:
            return [name, "—", "—", "—", "—", "—", "—", note]
        rmse  = rmse_override  if rmse_override  is not None else obj.get("RMSE",  None)
        mae   = mae_override   if mae_override   is not None else obj.get("MAE",   None)
        nrmse = nrmse_override if nrmse_override is not None else obj.get("nRMSE", None)
        mape  = mape_override  if mape_override  is not None else obj.get("MAPE",  None)
        mbe   = mbe_override   if mbe_override   is not None else obj.get("MBE",   None)
        r2    = r2_override    if r2_override    is not None else obj.get("R2",    None)
        return [
            name,
            fmt(rmse, 0, " W"),
            fmt(mae, 0, " W"),
            fmt(nrmse, 1, "%") if nrmse is not None else "—",
            fmt(mape, 1, "%") if mape is not None else "—",
            fmt(mbe, 0, " W") if mbe is not None else "—",
            fmt(r2, 4),
            note,
        ]

    header = [["Model", "RMSE", "MAE", "nRMSE", "MAPE*", "MBE", "R²", "Notes"]]
    rows = [
        row("CNN-LSTM (Keras)",
            None,
            rmse_override=cnn_rmse_w,
            mae_override=cnn_mae_w,
            nrmse_override=None,
            mape_override=54.4,
            mbe_override=None,
            r2_override=cnn["R2"] if cnn is not None else None,
            note="Best overall"),
        row("XGBoost DMS", xgb, note="Best tabular"),
        row("Hybrid (XGB+LSTM)", hyb, note="Matched XGB"),
        row("Same-day baseline", sday, note="Strong baseline"),
        row("LSTM pretrain+finetune", lstm, note="2-phase"),
        row("LSTM direct (synthetic)", lstm_s, note="Fast train"),
        row("Persistence baseline", pers, note="Lower bound"),
    ]

    data = header + rows
    col_w = [4.2*cm, 2.0*cm, 2.0*cm, 1.7*cm, 1.7*cm, 2.0*cm, 1.6*cm, 2.3*cm]
    t = make_table(data, col_w)

    # Highlight best (CNN-LSTM) and worst (Persistence)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 1), (-1, 1), BEST_COLOR),   # CNN-LSTM row
        ("BACKGROUND", (0, 7), (-1, 7), WORST_COLOR),  # Persistence row
    ]))
    return t


if __name__ == "__main__":
    build()
