"""Pure-Pillow publication plot renderer for SilentX.

Bypasses matplotlib native C DLL blocks by using Pillow directly to draw
clean, high-resolution diagnostic plots.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def draw_panel_electricity_accuracy(draw: ImageDraw.ImageDraw, df: pd.DataFrame, bbox: tuple):
    """Draw Panel 1: Chronological Accuracy Tracking (True Acc vs. Estimators)."""
    x0, y0, x1, y1 = bbox
    pad = 40
    plot_x0, plot_y0, plot_x1, plot_y1 = x0 + pad + 30, y0 + pad, x1 - pad, y1 - pad - 20

    # Draw border and grid
    draw.rectangle([plot_x0, plot_y0, plot_x1, plot_y1], outline="#CCCCCC", fill="#FAFAFA")

    # Gridlines
    for y_val in [0.4, 0.6, 0.8, 1.0]:
        y_pos = plot_y1 - (y_val - 0.3) / 0.7 * (plot_y1 - plot_y0)
        draw.line([(plot_x0, y_pos), (plot_x1, y_pos)], fill="#EAEAEA", width=1)
        draw.text((x0 + 10, y_pos - 6), f"{y_val:.1f}", fill="#555555")

    n_pts = len(df)
    xs = [plot_x0 + (i / max(1, n_pts - 1)) * (plot_x1 - plot_x0) for i in range(n_pts)]

    def to_y(val):
        return plot_y1 - (np.clip(val, 0.3, 1.0) - 0.3) / 0.7 * (plot_y1 - plot_y0)

    # Plot True Accuracy (Solid Black)
    true_pts = [(xs[i], to_y(df["true_accuracy"].iloc[i])) for i in range(n_pts)]
    draw.line(true_pts, fill="#000000", width=3)
    for pt in true_pts:
        draw.ellipse([pt[0] - 3, pt[1] - 3, pt[0] + 3, pt[1] + 3], fill="#000000")

    # Plot CBPE (Red Dashed)
    cbpe_pts = [(xs[i], to_y(df["est_cbpe"].iloc[i])) for i in range(n_pts)]
    draw.line(cbpe_pts, fill="#D9534F", width=2)

    # Plot ATC (Blue)
    atc_pts = [(xs[i], to_y(df["est_atc"].iloc[i])) for i in range(n_pts)]
    draw.line(atc_pts, fill="#337AB7", width=2)

    # Plot COTT (Green)
    cott_pts = [(xs[i], to_y(df["est_cott"].iloc[i])) for i in range(n_pts)]
    draw.line(cott_pts, fill="#5CB85C", width=2)

    # Panel Title & Legend
    draw.text((x0 + pad, y0 + 10), "OpenML Electricity: Chronological Accuracy Tracking", fill="#111111")
    leg_x = plot_x0 + 10
    leg_y = plot_y1 - 25
    draw.line([(leg_x, leg_y), (leg_x + 15, leg_y)], fill="#000000", width=3)
    draw.text((leg_x + 20, leg_y - 6), "True Acc", fill="#000000")

    draw.line([(leg_x + 85, leg_y), (leg_x + 100, leg_y)], fill="#D9534F", width=2)
    draw.text((leg_x + 105, leg_y - 6), "CBPE", fill="#D9534F")

    draw.line([(leg_x + 160, leg_y), (leg_x + 175, leg_y)], fill="#337AB7", width=2)
    draw.text((leg_x + 180, leg_y - 6), "ATC", fill="#337AB7")

    draw.line([(leg_x + 235, leg_y), (leg_x + 250, leg_y)], fill="#5CB85C", width=2)
    draw.text((leg_x + 255, leg_y - 6), "COTT", fill="#5CB85C")


def draw_panel_electricity_drift(draw: ImageDraw.ImageDraw, df: pd.DataFrame, bbox: tuple):
    """Draw Panel 2: TrustGuard Risk vs. Traditional Input Drift."""
    x0, y0, x1, y1 = bbox
    pad = 40
    plot_x0, plot_y0, plot_x1, plot_y1 = x0 + pad + 30, y0 + pad, x1 - pad, y1 - pad - 20

    draw.rectangle([plot_x0, plot_y0, plot_x1, plot_y1], outline="#CCCCCC", fill="#FAFAFA")

    # Gridlines
    for y_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y_pos = plot_y1 - (y_val / 1.0) * (plot_y1 - plot_y0)
        draw.line([(plot_x0, y_pos), (plot_x1, y_pos)], fill="#EAEAEA", width=1)
        draw.text((x0 + 10, y_pos - 6), f"{y_val:.2f}", fill="#555555")

    n_pts = len(df)
    xs = [plot_x0 + (i / max(1, n_pts - 1)) * (plot_x1 - plot_x0) for i in range(n_pts)]

    def to_y(val):
        return plot_y1 - np.clip(val, 0.0, 1.0) * (plot_y1 - plot_y0)

    # Plot TrustGuard Risk (Purple)
    tg_pts = [(xs[i], to_y(df["tg_risk"].iloc[i])) for i in range(n_pts)]
    draw.line(tg_pts, fill="#800080", width=3)

    # Plot KS_max (Orange)
    ks_pts = [(xs[i], to_y(df["ks_max"].iloc[i])) for i in range(n_pts)]
    draw.line(ks_pts, fill="#FFA500", width=2)

    # Plot PSI_max / 2.0 (Brown)
    psi_pts = [(xs[i], to_y(df["psi_max"].iloc[i] / 2.0)) for i in range(n_pts)]
    draw.line(psi_pts, fill="#A52A2A", width=2)

    draw.text((x0 + pad, y0 + 10), "OpenML Electricity: TrustGuard Risk vs. Traditional Input Drift", fill="#111111")
    leg_x = plot_x0 + 10
    leg_y = plot_y0 + 15
    draw.line([(leg_x, leg_y), (leg_x + 15, leg_y)], fill="#800080", width=3)
    draw.text((leg_x + 20, leg_y - 6), "TrustGuard Risk", fill="#800080")

    draw.line([(leg_x + 130, leg_y), (leg_x + 145, leg_y)], fill="#FFA500", width=2)
    draw.text((leg_x + 150, leg_y - 6), "KS_max", fill="#FFA500")

    draw.line([(leg_x + 220, leg_y), (leg_x + 235, leg_y)], fill="#A52A2A", width=2)
    draw.text((leg_x + 240, leg_y - 6), "PSI_max / 2", fill="#A52A2A")


def draw_panel_mae_barchart(draw: ImageDraw.ImageDraw, agg_df: pd.DataFrame, bbox: tuple):
    """Draw Panel 3: Estimation MAE Comparison Bar Chart."""
    x0, y0, x1, y1 = bbox
    pad = 40
    plot_x0, plot_y0, plot_x1, plot_y1 = x0 + pad + 30, y0 + pad, x1 - pad, y1 - pad - 20

    draw.rectangle([plot_x0, plot_y0, plot_x1, plot_y1], outline="#CCCCCC", fill="#FAFAFA")

    # Gridlines
    for y_val in [0.0, 0.05, 0.10, 0.15, 0.20]:
        y_pos = plot_y1 - (y_val / 0.22) * (plot_y1 - plot_y0)
        draw.line([(plot_x0, y_pos), (plot_x1, y_pos)], fill="#EAEAEA", width=1)
        draw.text((x0 + 10, y_pos - 6), f"{y_val:.2f}", fill="#555555")

    methods = ["AC", "DoC", "ATC", "CBPE", "COTT"]
    maes = [
        agg_df["mae_ac"].iloc[0],
        agg_df["mae_doc"].iloc[0],
        agg_df["mae_atc"].iloc[0],
        agg_df["mae_cbpe"].iloc[0],
        agg_df["mae_cott"].iloc[0],
    ]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#2CA02C"]

    n_bars = len(methods)
    bar_w = 40
    gap = ((plot_x1 - plot_x0) - (n_bars * bar_w)) / (n_bars + 1)

    for i in range(n_bars):
        bx0 = plot_x0 + gap + i * (bar_w + gap)
        bx1 = bx0 + bar_w
        by1 = plot_y1
        by0 = plot_y1 - (maes[i] / 0.22) * (plot_y1 - plot_y0)

        draw.rectangle([bx0, by0, bx1, by1], fill=colors[i], outline="#333333")
        draw.text((bx0 + 6, plot_y1 + 5), methods[i], fill="#111111")
        draw.text((bx0 - 2, by0 - 15), f"{maes[i]:.3f}", fill="#111111")

    draw.text((x0 + pad, y0 + 10), "Accuracy Estimation MAE on Electricity Stream (Lower is Better)", fill="#111111")


def draw_panel_synthetic_regimes(draw: ImageDraw.ImageDraw, synth_df: pd.DataFrame, bbox: tuple):
    """Draw Panel 4: Synthetic 6-Regime Stress Test Performance."""
    x0, y0, x1, y1 = bbox
    pad = 40
    plot_x0, plot_y0, plot_x1, plot_y1 = x0 + pad + 30, y0 + pad, x1 - pad, y1 - pad - 20

    draw.rectangle([plot_x0, plot_y0, plot_x1, plot_y1], outline="#CCCCCC", fill="#FAFAFA")

    # Gridlines
    for y_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y_pos = plot_y1 - (y_val / 1.0) * (plot_y1 - plot_y0)
        draw.line([(plot_x0, y_pos), (plot_x1, y_pos)], fill="#EAEAEA", width=1)
        draw.text((x0 + 10, y_pos - 6), f"{y_val:.2f}", fill="#555555")

    regimes_order = ["in_dist", "harmless", "corr_coll", "prior_sh", "concept_sh", "conf_coll"]
    regime_names = ["in_distribution", "harmless_drift", "correlation_collapse", "prior_shift", "concept_shift", "confidence_collapse"]

    means = synth_df.groupby("regime")[["true_accuracy", "est_cbpe", "tg_risk"]].mean()

    n_groups = len(regime_names)
    group_w = (plot_x1 - plot_x0) / n_groups
    sub_w = 12

    for i, r_name in enumerate(regime_names):
        if r_name not in means.index:
            continue
        gx = plot_x0 + i * group_w + (group_w - 3 * sub_w) / 2.0

        t_acc = means.loc[r_name, "true_accuracy"]
        c_acc = means.loc[r_name, "est_cbpe"]
        tg_r = means.loc[r_name, "tg_risk"]

        # Bar 1: True Acc (Black)
        draw.rectangle([gx, plot_y1 - t_acc * (plot_y1 - plot_y0), gx + sub_w, plot_y1], fill="#000000")
        # Bar 2: CBPE (Red)
        draw.rectangle([gx + sub_w, plot_y1 - c_acc * (plot_y1 - plot_y0), gx + 2 * sub_w, plot_y1], fill="#D9534F")
        # Bar 3: TG Risk (Purple)
        draw.rectangle([gx + 2 * sub_w, plot_y1 - tg_r * (plot_y1 - plot_y0), gx + 3 * sub_w, plot_y1], fill="#800080")

        draw.text((gx - 5, plot_y1 + 5), regimes_order[i], fill="#222222")

    draw.text((x0 + pad, y0 + 10), "Synthetic 6-Regime Audit: Model Collapse vs. Monitors", fill="#111111")
    leg_x = plot_x0 + 10
    leg_y = plot_y0 + 15
    draw.rectangle([leg_x, leg_y, leg_x + 10, leg_y + 10], fill="#000000")
    draw.text((leg_x + 15, leg_y - 2), "True Acc", fill="#000000")

    draw.rectangle([leg_x + 85, leg_y, leg_x + 95, leg_y + 10], fill="#D9534F")
    draw.text((leg_x + 100, leg_y - 2), "CBPE", fill="#D9534F")

    draw.rectangle([leg_x + 155, leg_y, leg_x + 165, leg_y + 10], fill="#800080")
    draw.text((leg_x + 170, leg_y - 2), "TG Risk", fill="#800080")


def generate_stage2_plots(
    elec_windows_csv: str = "results/stage2_electricity_windows.csv",
    elec_agg_csv: str = "results/stage2_electricity_aggregate.csv",
    synth_summary_csv: str = "results/stage2_synthetic_summary.csv",
    output_png: str = "results/stage2_monitor_comparison.png",
):
    """Render 4-panel diagnostic comparison plot using Pillow."""
    elec_df = pd.read_csv(elec_windows_csv)
    agg_df = pd.read_csv(elec_agg_csv)
    synth_df = pd.read_csv(synth_summary_csv)

    img_w, img_h = 1400, 1000
    img = Image.new("RGB", (img_w, img_h), color="#FFFFFF")
    draw = ImageDraw.Draw(img)

    # 4 quadrants: (0, 0), (w/2, 0), (0, h/2), (w/2, h/2)
    draw_panel_electricity_accuracy(draw, elec_df, bbox=(0, 0, img_w // 2, img_h // 2))
    draw_panel_electricity_drift(draw, elec_df, bbox=(img_w // 2, 0, img_w, img_h // 2))
    draw_panel_mae_barchart(draw, agg_df, bbox=(0, img_h // 2, img_w // 2, img_h))
    draw_panel_synthetic_regimes(draw, synth_df, bbox=(img_w // 2, img_h // 2, img_w, img_h))

    # Divider lines
    draw.line([(0, img_h // 2), (img_w, img_h // 2)], fill="#DDDDDD", width=2)
    draw.line([(img_w // 2, 0), (img_w // 2, img_h)], fill="#DDDDDD", width=2)

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    img.save(output_png)
    print(f"[+] Publication diagnostic plot saved to: {output_png}")


if __name__ == "__main__":
    generate_stage2_plots()
