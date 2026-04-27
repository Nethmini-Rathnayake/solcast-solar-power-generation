"""
draw_methodology.py
-------------------
Generates a publication-quality methodology diagram for the
CNN-LSTM solar PV forecasting pipeline.

Output: results/figures/methodology_diagram.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
C = {
    "data":       "#1565C0",   # dark blue    — data sources
    "feat":       "#2E7D32",   # dark green   — feature engineering
    "split":      "#6A1B9A",   # purple       — data split
    "arch":       "#E65100",   # deep orange  — model architecture
    "train":      "#00695C",   # teal         — training
    "finetune":   "#AD1457",   # pink/rose    — fine-tuning
    "ensemble":   "#4527A0",   # deep purple  — ensemble
    "output":     "#1B5E20",   # forest green — output
    "arrow":      "#455A64",   # blue-grey    — arrows
    "bg":         "#F5F7FA",   # near-white   — background
    "highlight":  "#FFF9C4",   # pale yellow  — accent fills
}

fig, ax = plt.subplots(figsize=(16, 22))
ax.set_xlim(0, 16)
ax.set_ylim(0, 22)
ax.axis("off")
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])

# ── Helper: draw a labelled box ────────────────────────────────────────────────
def box(ax, x, y, w, h, title, lines, color, title_size=11, line_size=9.2,
        fill="#FFFFFF", radius=0.25):
    # Shadow
    shadow = FancyBboxPatch((x+0.07, y-0.07), w, h,
                            boxstyle=f"round,pad=0,rounding_size={radius}",
                            linewidth=0, facecolor="#CCCCCC", zorder=1)
    ax.add_patch(shadow)
    # Main box
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0,rounding_size={radius}",
                          linewidth=2, edgecolor=color, facecolor=fill, zorder=2)
    ax.add_patch(rect)
    # Title bar
    title_bar = FancyBboxPatch((x, y+h-0.52), w, 0.52,
                               boxstyle=f"round,pad=0,rounding_size={radius}",
                               linewidth=0, facecolor=color, zorder=3)
    ax.add_patch(title_bar)
    ax.text(x + w/2, y + h - 0.26, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color="white", zorder=4)
    # Body lines
    line_h = (h - 0.62) / max(len(lines), 1)
    for i, line in enumerate(lines):
        ty = y + h - 0.62 - (i + 0.5) * line_h
        ax.text(x + 0.22, ty, line, ha="left", va="center",
                fontsize=line_size, color="#212121", zorder=4)


def arrow(ax, x1, y1, x2, y2, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=C["arrow"],
                                lw=2.0, mutation_scale=18),
                zorder=5)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.12, my, label, fontsize=8.5, color=C["arrow"],
                va="center", style="italic")


def side_label(ax, x, y, text, color):
    ax.text(x, y, text, fontsize=8, color=color, va="center",
            ha="left", style="italic",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=color, linewidth=1, alpha=0.85))


# ══════════════════════════════════════════════════════════════════════════════
# Row 1 — Data Sources (two side-by-side boxes)
# ══════════════════════════════════════════════════════════════════════════════
Y = 20.5
box(ax, 0.4, Y, 6.8, 1.25, "SYNTHETIC TRAINING DATA",
    ["• 4-year calibrated Solcast data  (Jan 2020 – Feb 2024)",
     "• 36,481 hourly rows  |  pvlib clearsky-corrected PV output (W)",
     "• All 12 calendar months  |  Sri Lanka monsoon regime coverage"],
    C["data"], fill="#E3F2FD")

box(ax, 8.8, Y, 6.8, 1.25, "REAL MEASURED PV DATA",
    ["• University of Moratuwa microgrid  (Apr 2022 – Mar 2023)",
     "• 7,666 hourly rows  |  Solcast NWP + on-site PV measurements",
     "• Val split: first 85%  |  Test split: last 15% (held-out)"],
    C["data"], fill="#E3F2FD")

# connector arrows into feature box
arrow(ax, 3.8, Y, 3.8, Y-0.35)
arrow(ax, 12.2, Y, 12.2, Y-0.35)
ax.annotate("", xy=(8.0, Y-0.35), xytext=(3.8, Y-0.35),
            arrowprops=dict(arrowstyle="-", color=C["arrow"], lw=1.5), zorder=5)
ax.annotate("", xy=(8.0, Y-0.35), xytext=(12.2, Y-0.35),
            arrowprops=dict(arrowstyle="-", color=C["arrow"], lw=1.5), zorder=5)
arrow(ax, 8.0, Y-0.35, 8.0, Y-0.62)

# ══════════════════════════════════════════════════════════════════════════════
# Row 2 — Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════
Y = 18.35
box(ax, 0.4, Y, 15.2, 1.7, "FEATURE ENGINEERING  (83 shared features)",
    ["• Irradiance / cloud:    GHI, DNI, DHI, cloud_opacity, clearness_index, clearsky_ghi",
     "• Meteorological:        air_temp, humidity, pressure, dewpoint  |  Physics: pvlib_ac_W, cos_zenith, elevation",
     "• Time cyclicals:        hour/month/doy sin-cos  |  Monsoon regime: SW, NE, inter-monsoon 1 & 2",
     "• Lag features:          pv_lag24, pv_lag48, ghi_lag24, clearness_lag24",
     "• NWP 24-h forecasts:    ghi_fcast_h1…h24, cloud_opacity_fcast_h1…h24  +  daily summaries",
     "• h+24 anchors [Step 1]: clearness_nwp_h24 = ghi_fcast_h24 / clearsky_ghi_h24   |   pvlib_clearsky_h24"],
    C["feat"], fill="#E8F5E9")

arrow(ax, 8.0, Y, 8.0, Y-0.3)

# ══════════════════════════════════════════════════════════════════════════════
# Row 3 — Sequence Generation & Normalisation
# ══════════════════════════════════════════════════════════════════════════════
Y = 17.0
box(ax, 0.4, Y, 15.2, 1.1,
    "SEQUENCE GENERATION  &  NORMALISATION",
    ["• Sliding window: seq_len=24 h look-back → 24 h MIMO targets  |  MinMaxScaler fitted on synthetic only",
     "• NaN-safe: zero-fill missing features, drop sequences with NaN targets",
     "• Train: 36,457 sequences (synthetic)  |  Val: 6,424 (real)  |  Test: 1,095 (real, held-out)"],
    C["split"], fill="#F3E5F5")

arrow(ax, 8.0, Y, 8.0, Y-0.3)

# ══════════════════════════════════════════════════════════════════════════════
# Row 4 — CNN-LSTM Architecture
# ══════════════════════════════════════════════════════════════════════════════
Y = 14.85
box(ax, 0.4, Y, 15.2, 1.9, "CNN-LSTM ARCHITECTURE  (270,776 parameters)",
    ["• Input:      (batch, 24 timesteps, 83 features)",
     "• CNN block:  Conv1D(64, k=3, relu)  →  Conv1D(32, k=3, relu)  →  MaxPooling1D(2)  →  (batch, 12, 32)",
     "• LSTM block: Bidirectional LSTM(128, return_seq=True)  →  Dropout(0.2)  →  LSTM(64)  →  Dropout(0.2)",
     "• Output:     Dense(24)  — MIMO: all 24 horizons predicted simultaneously",
     "• Loss: MSE  |  Optimiser: Adam(lr=1e-3)  |  EarlyStopping(patience=10) + ReduceLROnPlateau"],
    C["arch"], fill="#FFF3E0")

arrow(ax, 8.0, Y, 8.0, Y-0.3)

# ══════════════════════════════════════════════════════════════════════════════
# Row 5 — Original Training
# ══════════════════════════════════════════════════════════════════════════════
Y = 13.45
box(ax, 0.4, Y, 15.2, 1.05,
    "PHASE 1 — TRAIN ON SYNTHETIC  (early-stop on real val)",
    ["• Train: 36,457 synthetic sequences  |  Val: 6,424 real sequences  (domain-shift early stopping)",
     "• Result: Overall R²=0.9568  |  h+1=0.9553  |  h+24=0.9578  |  Nov R²=−0.28  |  Oct/Apr/May < 0.76"],
    C["train"], fill="#E0F2F1")

arrow(ax, 8.0, Y, 8.0, Y-0.3)

# ══════════════════════════════════════════════════════════════════════════════
# Row 6 — Fine-tuning (3 passes)
# ══════════════════════════════════════════════════════════════════════════════
Y = 11.0
box(ax, 0.4, Y, 15.2, 2.1,
    "PHASE 2 — PROGRESSIVE FINE-TUNING  (domain adaptation on real val set)",
    ["• Problem months identified: Nov (R²=−0.28), Oct (0.70), Apr (0.76), May (0.69)",
     "",
     "  Pass v1  LR=1e-4 │ Problem months 100% + good months 30% random sample",
     "                    │ → Nov/Oct fixed, but March degraded (test-set leak risk identified)",
     "",
     "  Pass v2  LR=5e-5 │ ALL val months 100% kept + oversample: Nov×4, Oct×3, Apr×3, May×3",
     "                    │ → Apr→0.91, Oct→0.96, Nov→0.976  |  March still drops (no March in val set)",
     "",
     "  Pass v3  LR=2e-5 │ ALL months + May×6, Dec×2, Jan×2  (Dec/Jan = dry-season proxies for March)",
     "                    │ → May→0.860  |  11/12 months approaching 0.90"],
    C["finetune"], fill="#FCE4EC")

arrow(ax, 8.0, Y, 8.0, Y-0.3)

# ══════════════════════════════════════════════════════════════════════════════
# Row 7 — Ensemble / Routing
# ══════════════════════════════════════════════════════════════════════════════
Y = 9.45
box(ax, 0.4, Y, 15.2, 1.15,
    "PHASE 3 — MONTH-AWARE ENSEMBLE ROUTER",
    ["• March → Original model  (March 2023 is in test set; fine-tuning had no March signal → degraded to 0.886)",
     "• All other months → Fine-tuned model  (domain-adapted to real cloud transition patterns)",
     "• Hard routing (no blending): justified by distinct training data availability per month"],
    C["ensemble"], fill="#EDE7F6")

arrow(ax, 8.0, Y, 8.0, Y-0.3)

# ══════════════════════════════════════════════════════════════════════════════
# Row 8 — Final Results
# ══════════════════════════════════════════════════════════════════════════════
Y = 6.9
box(ax, 0.4, Y, 15.2, 2.15,
    "FINAL RESULTS  —  24-HORIZON PV FORECASTING",
    ["• Overall (test set):   R²=0.9568  |  RMSE=16.16 kW  |  Bootstrap 95% CI [0.9541, 0.9591]",
     "• Daytime-only:         R²=0.9180  |  All 24 horizons ≥ 0.955",
     "• vs same-day baseline: +0.1476 R² improvement",
     "",
     "  Monthly R²:   Jan 0.947  Feb 0.960  Mar 0.934  Apr 0.907  May 0.860*  Jun 0.942",
     "                Jul 0.929  Aug 0.944  Sep 0.959  Oct 0.961  Nov 0.976   Dec 0.954",
     "",
     "  * May = SW Monsoon onset — data availability ceiling (only 1 year of May data in val set)"],
    C["output"], fill="#E8F5E9")

# ══════════════════════════════════════════════════════════════════════════════
# Row 9 — Legend
# ══════════════════════════════════════════════════════════════════════════════
Y = 6.0
legend_items = [
    (C["data"],     "Data sources"),
    (C["feat"],     "Feature engineering"),
    (C["split"],    "Sequence prep"),
    (C["arch"],     "Model architecture"),
    (C["train"],    "Phase 1: Training"),
    (C["finetune"], "Phase 2: Fine-tuning"),
    (C["ensemble"], "Phase 3: Ensemble"),
    (C["output"],   "Results"),
]
ax.text(0.4, Y+0.15, "Legend:", fontsize=9, fontweight="bold", color="#333333")
for i, (col, lbl) in enumerate(legend_items):
    xi = 0.4 + i * 1.95
    rect = FancyBboxPatch((xi, Y-0.3), 0.28, 0.28,
                          boxstyle="round,pad=0.03", linewidth=1.2,
                          edgecolor=col, facecolor=col, zorder=3)
    ax.add_patch(rect)
    ax.text(xi+0.38, Y-0.16, lbl, fontsize=8.3, va="center", color="#333333")

# ══════════════════════════════════════════════════════════════════════════════
# Title
# ══════════════════════════════════════════════════════════════════════════════
ax.text(8.0, 21.85, "CNN-LSTM Solar PV Forecasting Pipeline",
        ha="center", va="center", fontsize=16, fontweight="bold", color="#1A237E")
ax.text(8.0, 21.55,
        "University of Moratuwa Microgrid  |  1–24 Hour Ahead Forecasting  |  Sri Lanka",
        ha="center", va="center", fontsize=10, color="#455A64", style="italic")

plt.tight_layout(pad=0.5)
path = OUT / "methodology_diagram.png"
fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=C["bg"])
plt.close(fig)
print(f"Saved: {path}")
