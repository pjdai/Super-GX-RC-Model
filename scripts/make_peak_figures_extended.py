"""Build the multi-horizon three-day figure and the peak-performance summary.

Both figures now overlay anchor-stale and propagating rollouts so the two
modes can be visually compared on the same panels.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT = REPO_ROOT / "Room_Temp_Rolling" / "output" / "rolling_predictions_vs_actual.xlsx"
OUTDIR = INPUT.parent

PEAK_START, PEAK_END = 14, 18

# Color scheme: 1h darkest blue → 24h lightest blue (Panel A)
HORIZON_BLUES = {
    "1h Gated": "#08306b",
    "6h":       "#2171b5",
    "12h":      "#6baed6",
    "24h":      "#c6dbef",
}
# Propagating uses the purple palette so anchor vs prop is visually distinct
PROP_PURPLES = {
    "6h":  "#54278f",
    "12h": "#807dba",
    "24h": "#bcbddc",
}
# Per-horizon line color used in the combined overlay panel
ROLLOUT_COLORS = {
    "6h":  "#2ca02c",   # green
    "12h": "#ff7f0e",   # orange
    "24h": "#d62728",   # red
}

plt.rcParams.update({
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "font.family": "DejaVu Sans",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Load + derive
# ---------------------------------------------------------------------------
df = pd.read_excel(INPUT)
df["Date"] = pd.to_datetime(df["Date"])
df["hour"] = df["Date"].dt.hour
df["dow"] = df["Date"].dt.dayofweek
df["day"] = df["Date"].dt.normalize()

ROLLOUT_COLS = ["T_rollout_3h (F)", "T_rollout_6h (F)", "T_rollout_12h (F)", "T_rollout_24h (F)"]
PROP_COLS    = ["T_rollout_3h_prop (F)", "T_rollout_6h_prop (F)",
                "T_rollout_12h_prop (F)", "T_rollout_24h_prop (F)"]
GATED_COL = "T_full_gated (F)"
HAS_PROP = all(c in df.columns for c in PROP_COLS)
print(f"Has propagating columns: {HAS_PROP}")

cols_for_err = ROLLOUT_COLS + [GATED_COL] + (PROP_COLS if HAS_PROP else [])
for c in cols_for_err:
    df[f"err_{c}"] = df["Room Temp (F)"] - df[c]
    df[f"abserr_{c}"] = df[f"err_{c}"].abs()

peak_mask = (df["hour"] >= PEAK_START) & (df["hour"] <= PEAK_END)
peak = df[peak_mask].copy()

daily_mae = peak.groupby("day").agg(
    {f"abserr_{c}": "mean" for c in cols_for_err}
).rename(columns={f"abserr_{c}": c for c in cols_for_err})

# Day selectors
peak_idx = peak["Room Temp (F)"].idxmax()
peak_day = pd.Timestamp(peak.loc[peak_idx, "day"])

mean_mae_4_horizons = daily_mae[ROLLOUT_COLS].mean(axis=1)
median_overall = mean_mae_4_horizons.median()
avg_day = pd.Timestamp((mean_mae_4_horizons - median_overall).abs().idxmin())

_worst_ranked = daily_mae["T_rollout_24h (F)"].sort_values(ascending=False)
worst_day = pd.Timestamp(_worst_ranked.index[0])
if worst_day == peak_day and len(_worst_ranked) > 1:
    print(f"  [note] worst day (24h-MAE) coincides with peak day {peak_day.date()}; "
          f"using runner-up: {_worst_ranked.index[1].date()} "
          f"(24h MAE = {_worst_ranked.iloc[1]:.3f}°F)")
    worst_day = pd.Timestamp(_worst_ranked.index[1])

print(f"Peak day:    {peak_day.date()}  (max actual = {peak.loc[peak_idx, 'Room Temp (F)']:.2f}°F)")
print(f"Average day: {avg_day.date()}  (4-horizon mean MAE = {mean_mae_4_horizons[avg_day]:.3f}°F, "
      f"overall median = {median_overall:.3f}°F)")
print(f"Worst day:   {worst_day.date()}  (24h MAE = {daily_mae.loc[worst_day, 'T_rollout_24h (F)']:.3f}°F)")
print(f"Total unique days: {len(daily_mae)}")


def day_slice(target: pd.Timestamp) -> pd.DataFrame:
    return df[df["day"] == target].sort_values("Date").reset_index(drop=True)


# ===========================================================================
# FIGURE 1: 4 rows × 3 cols (horizons × days), anchor + propagating overlays
# ===========================================================================
days = [("Peak Day", peak_day), ("Average Day", avg_day), ("Worst Day", worst_day)]
horizons_rows = [
    ("6h Rollout",  "T_rollout_6h (F)",  "T_rollout_6h_prop (F)",  "6h"),
    ("12h Rollout", "T_rollout_12h (F)", "T_rollout_12h_prop (F)", "12h"),
    ("24h Rollout", "T_rollout_24h (F)", "T_rollout_24h_prop (F)", "24h"),
]
day_frames = {label: day_slice(d) for label, d in days}

fig1, axes = plt.subplots(4, 3, figsize=(18, 16), sharex=True)
fig1.suptitle(
    "Multi-Horizon Forecast Comparison — Peak / Average / Worst Days "
    "(anchor-stale  vs  propagating)",
    fontsize=15, y=0.995,
)

for r, (row_label, anchor_col, prop_col, hkey) in enumerate(horizons_rows):
    all_vals = []
    for df_d in day_frames.values():
        all_vals.append(df_d["Room Temp (F)"].values)
        all_vals.append(df_d[anchor_col].values)
        if HAS_PROP:
            all_vals.append(df_d[prop_col].values)
    ymin = float(np.nanmin(np.concatenate(all_vals)))
    ymax = float(np.nanmax(np.concatenate(all_vals)))
    pad = (ymax - ymin) * 0.10 + 0.5

    for c, (col_title, day_ts) in enumerate(days):
        ax = axes[r, c]
        d_df = day_frames[col_title]
        hours = d_df["hour"].values
        actual = d_df["Room Temp (F)"].values
        pred_a = d_df[anchor_col].values
        pred_p = d_df[prop_col].values if HAS_PROP else None

        ax.axvspan(PEAK_START, PEAK_END, color="#ffb347", alpha=0.30, zorder=0)
        ax.plot(hours, actual, "o-", color="#1f77b4", lw=2, ms=5, label="Actual")
        ax.plot(hours, pred_a, "s--", color="#d62728", lw=1.8, ms=5,
                label=f"{hkey} anchor")
        if HAS_PROP:
            ax.plot(hours, pred_p, "^:", color="#54278f", lw=1.8, ms=5,
                    label=f"{hkey} prop")
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_xlim(-0.5, 23.5)
        ax.set_xticks(range(0, 24, 3))

        # Peak-window MAEs annotated inside the orange band
        in_pk = (hours >= PEAK_START) & (hours <= PEAK_END)
        mae_a = float(np.mean(np.abs(actual[in_pk] - pred_a[in_pk])))
        if HAS_PROP:
            mae_p = float(np.mean(np.abs(actual[in_pk] - pred_p[in_pk])))
            ann = f"Peak MAE\nanchor: {mae_a:.2f}°F\nprop:   {mae_p:.2f}°F"
        else:
            ann = f"Peak MAE: {mae_a:.2f}°F"
        ax.annotate(
            ann,
            xy=((PEAK_START + PEAK_END) / 2, ax.get_ylim()[1]),
            xytext=(0, -10),
            textcoords="offset points",
            ha="center", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ffb347", lw=1.0),
        )

        if r == 0:
            ax.set_title(f"{col_title} ({day_ts.date()})", fontsize=13)
        if c == 0:
            ax.set_ylabel(f"{row_label}\n\nRoom Temp (°F)", fontsize=11)
        if r == 0 and c == 2:
            ax.legend(loc="upper right", framealpha=0.9, fontsize=8)

# Row 3: combined overlay (all rollouts, both modes)
all_vals_combined = []
for df_d in day_frames.values():
    all_vals_combined.append(df_d["Room Temp (F)"].values)
    for _, anchor_col, prop_col, _ in horizons_rows:
        all_vals_combined.append(df_d[anchor_col].values)
        if HAS_PROP:
            all_vals_combined.append(df_d[prop_col].values)
ymin_c = float(np.nanmin(np.concatenate(all_vals_combined)))
ymax_c = float(np.nanmax(np.concatenate(all_vals_combined)))
pad_c = (ymax_c - ymin_c) * 0.10 + 0.5

for c, (col_title, day_ts) in enumerate(days):
    ax = axes[3, c]
    d_df = day_frames[col_title]
    hours = d_df["hour"].values
    ax.axvspan(PEAK_START, PEAK_END, color="#ffb347", alpha=0.30, zorder=0)
    ax.plot(hours, d_df["Room Temp (F)"].values, "-", color="#1f77b4",
            lw=2.4, label="Actual", zorder=5)
    for hkey, anchor_col, prop_col in [
        ("6h",  "T_rollout_6h (F)",  "T_rollout_6h_prop (F)"),
        ("12h", "T_rollout_12h (F)", "T_rollout_12h_prop (F)"),
        ("24h", "T_rollout_24h (F)", "T_rollout_24h_prop (F)"),
    ]:
        ax.plot(hours, d_df[anchor_col].values, "--", lw=1.6,
                color=ROLLOUT_COLORS[hkey], label=f"{hkey} anchor")
        if HAS_PROP:
            ax.plot(hours, d_df[prop_col].values, ":", lw=1.6,
                    color=ROLLOUT_COLORS[hkey], label=f"{hkey} prop")
    ax.set_ylim(ymin_c - pad_c, ymax_c + pad_c)
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Hour of Day")
    if c == 0:
        ax.set_ylabel("All Horizons\n\nRoom Temp (°F)", fontsize=11)
    if c == 2:
        ax.legend(loc="upper right", framealpha=0.9, ncol=2, fontsize=8)

fig1.tight_layout(rect=(0, 0, 1, 0.985))
out1 = OUTDIR / "multi_horizon_three_days.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"Saved: {out1}")


# ===========================================================================
# FIGURE 2: 2x2 panel summary (anchor and propagating)
# ===========================================================================
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 14))
fig2.suptitle(
    f"Peak-Hour ({PEAK_START:02d}:00–{PEAK_END:02d}:00) Forecast Performance Summary  "
    f"({len(daily_mae)} days)  —  anchor vs propagating",
    fontsize=15, y=0.995,
)

# ---------- Panel A: paired error box plot per horizon ----------
axA = axes2[0, 0]
# Build (label, color, error array) tuples interleaved per horizon
boxA = [("1h Gated", HORIZON_BLUES["1h Gated"],
         peak[f"err_{GATED_COL}"].dropna().to_numpy())]
for hkey, anchor_col, prop_col in [
    ("6h",  "T_rollout_6h (F)",  "T_rollout_6h_prop (F)"),
    ("12h", "T_rollout_12h (F)", "T_rollout_12h_prop (F)"),
    ("24h", "T_rollout_24h (F)", "T_rollout_24h_prop (F)"),
]:
    boxA.append((f"{hkey}\nanchor", HORIZON_BLUES[hkey],
                 peak[f"err_{anchor_col}"].dropna().to_numpy()))
    if HAS_PROP:
        boxA.append((f"{hkey}\nprop", PROP_PURPLES[hkey],
                     peak[f"err_{prop_col}"].dropna().to_numpy()))

bp = axA.boxplot(
    [e for _, _, e in boxA],
    tick_labels=[lbl for lbl, _, _ in boxA],
    patch_artist=True, widths=0.55,
    medianprops=dict(color="black", lw=2),
    flierprops=dict(marker="o", markersize=4, markerfacecolor="gray",
                    markeredgecolor="gray", alpha=0.5),
)
for patch, (_, col, _) in zip(bp["boxes"], boxA):
    patch.set_facecolor(col)
    patch.set_edgecolor("black")
    patch.set_alpha(0.9)

axA.axhline(0, color="black", lw=1.2, ls="--")
axA.set_xlabel("Forecast Horizon  (anchor vs propagating)")
axA.set_ylabel("Error (°F)  —  actual − predicted")
axA.set_title("A. Peak-Hour Error Distribution by Horizon")
for i, (_, _, e) in enumerate(boxA, start=1):
    axA.annotate(f"μ={np.mean(e):+.2f}\nσ={np.std(e):.2f}",
                 xy=(i, axA.get_ylim()[1]), xytext=(0, -8),
                 textcoords="offset points", ha="center", va="top",
                 fontsize=8, color="#333")

# ---------- Panel B: daily peak-window MAE per horizon (anchor solid, prop dashed) ----------
axB = axes2[0, 1]
axB.plot(daily_mae.index, daily_mae[GATED_COL].values, "-o", lw=1.8, ms=4,
         color=HORIZON_BLUES["1h Gated"], label="1h Gated", alpha=0.95)
for hkey, anchor_col, prop_col in [
    ("6h",  "T_rollout_6h (F)",  "T_rollout_6h_prop (F)"),
    ("12h", "T_rollout_12h (F)", "T_rollout_12h_prop (F)"),
    ("24h", "T_rollout_24h (F)", "T_rollout_24h_prop (F)"),
]:
    axB.plot(daily_mae.index, daily_mae[anchor_col].values, "-o",
             lw=1.6, ms=3.5, color=HORIZON_BLUES[hkey],
             label=f"{hkey} anchor", alpha=0.95)
    if HAS_PROP:
        axB.plot(daily_mae.index, daily_mae[prop_col].values, "--^",
                 lw=1.6, ms=3.5, color=PROP_PURPLES[hkey],
                 label=f"{hkey} prop", alpha=0.95)
axB.axhline(1.0, color="black", lw=1.2, ls="--", alpha=0.7,
            label="MAE = 1.0°F threshold")
axB.axvline(peak_day, color="#ff7f0e", lw=1.5, ls=":", alpha=0.85)
axB.axvline(worst_day, color="#d62728", lw=1.5, ls=":", alpha=0.85)
ymax_B = float(daily_mae.drop(columns=[c for c in daily_mae.columns
                                       if c.startswith("T_phys")], errors="ignore"
                              ).max().max()) * 1.15
axB.text(peak_day, ymax_B * 0.98, f"Peak\n{peak_day.date()}",
         ha="center", va="top", fontsize=9, color="#ff7f0e",
         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#ff7f0e", lw=1.0))
axB.text(worst_day, ymax_B * 0.98, f"Worst\n{worst_day.date()}",
         ha="center", va="top", fontsize=9, color="#d62728",
         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#d62728", lw=1.0))
axB.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
axB.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
plt.setp(axB.get_xticklabels(), rotation=30, ha="right")
axB.set_xlabel("Date")
axB.set_ylabel(f"Peak-window MAE (°F)  —  {PEAK_START:02d}:00–{PEAK_END:02d}:00")
axB.set_title("B. Daily Peak-Hour MAE Over Time")
axB.legend(loc="upper left", ncol=3, framealpha=0.9, fontsize=8)

# ---------- Panel C: scatter of peak temp vs 24h error, anchor + propagating ----------
axC = axes2[1, 0]
daily_actual = peak.groupby("day")["Room Temp (F)"].mean()
daily_err24_a = peak.groupby("day")[f"err_T_rollout_24h (F)"].mean()
days_index = daily_err24_a.index
day_nums = mdates.date2num(days_index)
norm = Normalize(vmin=day_nums.min(), vmax=day_nums.max())

sc_a = axC.scatter(daily_actual.values, daily_err24_a.values,
                   c=day_nums, cmap="viridis", norm=norm,
                   marker="o", s=70, edgecolor="black", linewidth=0.5,
                   zorder=3, label="anchor")
if HAS_PROP:
    daily_err24_p = peak.groupby("day")[f"err_T_rollout_24h_prop (F)"].mean()
    sc_p = axC.scatter(daily_actual.values, daily_err24_p.values,
                       c=day_nums, cmap="viridis", norm=norm,
                       marker="^", s=70, edgecolor="black", linewidth=0.5,
                       zorder=3, label="propagating")

# Trend lines
def _trend(x, y, ax, color, label):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() >= 2:
        coef = np.polyfit(x[mask], y[mask], 1)
        xs = np.linspace(x[mask].min(), x[mask].max(), 50)
        ax.plot(xs, np.polyval(coef, xs), "-", color=color, lw=1.6,
                alpha=0.75, label=f"{label}: y={coef[0]:.2f}x{coef[1]:+.2f}")
_trend(daily_actual.values, daily_err24_a.values, axC, "#1f77b4", "anchor trend")
if HAS_PROP:
    _trend(daily_actual.values, daily_err24_p.values, axC, "#54278f", "prop trend")

axC.axhline(0, color="black", lw=1.2, ls="--", alpha=0.7)
for label, day_ts, color in [("Peak", peak_day, "#ff7f0e"),
                              ("Worst", worst_day, "#d62728")]:
    if day_ts in daily_actual.index:
        x0 = daily_actual.loc[day_ts]
        y0 = daily_err24_a.loc[day_ts]
        axC.annotate(
            f"{label}\n{day_ts.date()}",
            xy=(x0, y0), xytext=(15, 12), textcoords="offset points",
            fontsize=9, color=color,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1.0),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
        )

axC.set_xlabel("Mean actual Room Temp during peak window (°F)")
axC.set_ylabel("Mean 24h error (actual − predicted) during peak window (°F)")
axC.set_title("C. Peak-Hour Temperature vs. 24h Forecast Error")
axC.legend(loc="upper left", framealpha=0.9, fontsize=8)
cbar = fig2.colorbar(sc_a, ax=axC, fraction=0.046, pad=0.04)
cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
cbar.ax.set_ylabel("Date", fontsize=10)

# ---------- Panel D: difference heatmap (|anchor| − |prop|) by hour × day-of-week ----------
axD = axes2[1, 1]
abs_a = (df.assign(abserr=df["abserr_T_rollout_24h (F)"])
           .groupby(["hour", "dow"])["abserr"].mean()
           .unstack("dow")
           .reindex(index=range(24), columns=range(7)))
if HAS_PROP:
    abs_p = (df.assign(abserr=df["abserr_T_rollout_24h_prop (F)"])
               .groupby(["hour", "dow"])["abserr"].mean()
               .unstack("dow")
               .reindex(index=range(24), columns=range(7)))
    diff = abs_a.values - abs_p.values   # >0  ⇒  propagating better
    vmax = float(np.nanmax(np.abs(diff))) or 1.0
    norm_d = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = axD.imshow(diff, aspect="auto", origin="lower", cmap="RdBu",
                    norm=norm_d, extent=(-0.5, 6.5, -0.5, 23.5))
    panel_title = ("D. 24h |Error| Difference  (anchor − prop)\n"
                   "blue cells: propagating better;  red: anchor better")
    cb_label = "|anchor err| − |prop err|  (°F)"
else:
    im = axD.imshow(abs_a.values, aspect="auto", origin="lower", cmap="YlOrRd",
                    extent=(-0.5, 6.5, -0.5, 23.5))
    panel_title = "D. 24h Forecast |Error| by Hour × Day-of-Week"
    cb_label = "Mean |error| (°F)"

axD.set_xticks(range(7))
axD.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
axD.set_yticks(range(0, 24, 2))
axD.set_xlabel("Day of Week")
axD.set_ylabel("Hour of Day")
axD.set_title(panel_title)
axD.axhspan(PEAK_START - 0.5, PEAK_END + 0.5, facecolor="none",
            edgecolor="black", linestyle="--", linewidth=1.5)
axD.annotate(f"{PEAK_START:02d}–{PEAK_END:02d}",
             xy=(6.5, (PEAK_START + PEAK_END) / 2),
             xytext=(6, 2), textcoords="offset points",
             fontsize=9, ha="left", va="center",
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.8))
cb = fig2.colorbar(im, ax=axD, fraction=0.046, pad=0.04)
cb.set_label(cb_label, fontsize=10)

fig2.tight_layout(rect=(0, 0, 1, 0.985))
out2 = OUTDIR / "peak_performance_summary.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {out2}")
