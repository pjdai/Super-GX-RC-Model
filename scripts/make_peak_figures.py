"""Generate peak-day forecast and peak-hour error distribution figures."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT = REPO_ROOT / "Room_Temp_Rolling" / "output" / "rolling_predictions_vs_actual.xlsx"
OUTDIR = INPUT.parent

PEAK_START, PEAK_END = 14, 18  # inclusive hour range

plt.style.use("seaborn-v0_8-whitegrid")

df = pd.read_excel(INPUT)
df["Date"] = pd.to_datetime(df["Date"])
df["hour"] = df["Date"].dt.hour
df["day"] = df["Date"].dt.normalize()

# Anchor + propagating columns. _prop columns may be absent if the workbook
# was generated before propagating support landed; guard for that.
HORIZONS = ["6h", "12h", "24h"]
ROLLOUT_COLS_ANCHOR = [f"T_rollout_{h} (F)" for h in HORIZONS]
ROLLOUT_COLS_PROP   = [f"T_rollout_{h}_prop (F)" for h in HORIZONS]
HAS_PROP = all(c in df.columns for c in ROLLOUT_COLS_PROP)

print(f"Has propagating columns: {HAS_PROP}")

# ---------- Figure A: peak day 24h forecast ----------
peak_window = df[(df["hour"] >= PEAK_START) & (df["hour"] <= PEAK_END)]
peak_day = peak_window.loc[peak_window["Room Temp (F)"].idxmax(), "day"]

day_df = df[df["day"] == peak_day].sort_values("Date").reset_index(drop=True)
actual = day_df["Room Temp (F)"].to_numpy()
pred_anchor = day_df["T_rollout_24h (F)"].to_numpy()
hours = day_df["Date"].dt.hour.to_numpy()
err_anchor = actual - pred_anchor
pred_prop = day_df["T_rollout_24h_prop (F)"].to_numpy() if HAS_PROP else None
err_prop = (actual - pred_prop) if HAS_PROP else None

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(11, 7.6), sharex=True,
    gridspec_kw={"height_ratios": [3, 1.6], "hspace": 0.12},
)

ax1.axvspan(PEAK_START, PEAK_END, color="#ffb347", alpha=0.30,
            label=f"Peak demand window ({PEAK_START:02d}:00-{PEAK_END:02d}:00)")
ax1.plot(hours, actual, "o-", color="#1f77b4", lw=2.2, ms=5, label="Actual Room Temp")
ax1.plot(hours, pred_anchor, "s--", color="#d62728", lw=2, ms=5,
         label="T_rollout_24h (anchor-stale)")
if HAS_PROP:
    ax1.plot(hours, pred_prop, "^:", color="#9467bd", lw=2, ms=5,
             label="T_rollout_24h_prop (propagating)")
ax1.set_ylabel("Room Temperature (°F)")
ax1.set_title(f"Peak Day 24-Hour Forecast — {peak_day.date()} "
              f"(highest actual temp during {PEAK_START:02d}:00–{PEAK_END:02d}:00)")
ax1.legend(loc="best", framealpha=0.9)

# Bottom panel: paired error bars (anchor and propagating side by side per hour)
width = 0.40 if HAS_PROP else 0.85
offsets = (-0.20, 0.20) if HAS_PROP else (0.0,)
ax2.axhline(0, color="black", lw=0.8)
ax2.axvspan(PEAK_START, PEAK_END, color="#ffb347", alpha=0.30)
ax2.bar(hours + offsets[0], err_anchor, width=width,
        color=np.where(err_anchor >= 0, "#d62728", "#fcae91"),
        edgecolor="black", linewidth=0.4, alpha=0.85,
        label="Anchor error")
if HAS_PROP:
    ax2.bar(hours + offsets[1], err_prop, width=width,
            color=np.where(err_prop >= 0, "#54278f", "#bcbddc"),
            edgecolor="black", linewidth=0.4, alpha=0.85,
            label="Propagating error")
ax2.set_xlabel("Hour of Day")
ax2.set_ylabel("Error (°F)\nactual − predicted")
ax2.set_xticks(range(0, 24, 2))
ax2.set_xlim(-0.6, 23.6)
ax2.legend(loc="best", framealpha=0.9, ncol=2 if HAS_PROP else 1)

fig.savefig(OUTDIR / "peak_day_24h_forecast.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------- Figure B: peak-hour error distribution ----------
peak = df[(df["hour"] >= PEAK_START) & (df["hour"] <= PEAK_END)].copy()

# Build interleaved (anchor, propagating) box pairs per horizon.
errors_grouped = []
labels = []
colors = []
ANCHOR_PALETTE = {"6h": "#9ecae1", "12h": "#6baed6", "24h": "#3182bd"}
PROP_PALETTE   = {"6h": "#dadaeb", "12h": "#bcbddc", "24h": "#807dba"}
for h in HORIZONS:
    e_a = (peak["Room Temp (F)"] - peak[f"T_rollout_{h} (F)"]).dropna().to_numpy()
    errors_grouped.append(e_a)
    labels.append(f"{h}\nanchor")
    colors.append(ANCHOR_PALETTE[h])
    if HAS_PROP:
        e_p = (peak["Room Temp (F)"] - peak[f"T_rollout_{h}_prop (F)"]).dropna().to_numpy()
        errors_grouped.append(e_p)
        labels.append(f"{h}\nprop")
        colors.append(PROP_PALETTE[h])

fig, ax = plt.subplots(figsize=(11, 6.4))
bp = ax.boxplot(
    errors_grouped, tick_labels=labels, patch_artist=True, widths=0.6,
    medianprops=dict(color="black", lw=2),
    flierprops=dict(marker="o", markersize=4, markerfacecolor="gray",
                    markeredgecolor="gray", alpha=0.5),
)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_edgecolor("black")
    patch.set_alpha(0.9)

# Visual separators between horizon-pairs.
if HAS_PROP:
    for i in range(2, len(errors_grouped), 2):
        ax.axvline(i + 0.5, color="gray", lw=0.6, ls=":", alpha=0.5)

ax.axhline(0, color="black", lw=1.2, ls="--")
ax.set_xlabel("Forecast Horizon  (anchor-stale  |  propagating)")
ax.set_ylabel("Error (°F)  —  actual − predicted")
n_days = peak["day"].nunique()
n_per_box = len(errors_grouped[0])
ax.set_title(f"Peak-Hour ({PEAK_START:02d}:00–{PEAK_END:02d}:00) Forecast Error "
             f"Distribution\nacross {n_days} days  (n={n_per_box} hourly samples per box)")

for i, e in enumerate(errors_grouped, start=1):
    ax.annotate(f"μ={e.mean():+.2f}\nσ={e.std():.2f}",
                xy=(i, ax.get_ylim()[1]), xytext=(0, -8),
                textcoords="offset points", ha="center", va="top",
                fontsize=8, color="#333")

fig.savefig(OUTDIR / "peak_hour_error_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"Peak day selected: {peak_day.date()}")
print(f"  max actual in window: {peak_window.loc[peak_window['Room Temp (F)'].idxmax(), 'Room Temp (F)']:.2f} °F")
print(f"Days covered (peak window): {n_days}")
print(f"Saved: {OUTDIR / 'peak_day_24h_forecast.png'}")
print(f"Saved: {OUTDIR / 'peak_hour_error_distribution.png'}")
