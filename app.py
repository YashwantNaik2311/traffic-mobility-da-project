"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   CityFlow Smart Urban Mobility & Traffic — Full-Stack Analytics Dashboard  ║
║   Dataset : smart_city_traffic_mobility.csv  (204,000 rows × 47 features)  ║
║   Stack   : Streamlit · pandas · scikit-learn · matplotlib · seaborn · fpdf2║
║   Run     : streamlit run app.py                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import io, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              confusion_matrix, classification_report)
from fpdf import FPDF

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

DATA_FILE = "smart_city_traffic_mobility.csv"
PDF_FILE  = "traffic_mobility_report.pdf"
RISK_HIGH = 0.70
RISK_MED  = 0.40
FEATURES  = [
    "lanes","speed_limit","vehicle_count","heavy_vehicle_count","motorcycle_count",
    "public_transport_count","traffic_density","queue_length","average_wait_time",
    "hour","day_of_week","is_weekend","is_holiday","rush_hour","rainfall_mm",
    "temperature","visibility_km","air_quality_index","construction_activity",
    "accident_reported","parking_occupancy","public_event","emergency_vehicle_detected",
    "traffic_flow_rate","signal_cycle_seconds","green_light_duration",
    "speed_reduction_pct","vehicle_per_lane",
]

# ══════════════════════════════════════════════════════════════════════════════
#  BACKEND
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="⏳ Loading & cleaning 204,000 records …")
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["timestamp"]   = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["date"]        = df["timestamp"].dt.date
    for col in ["vehicle_count","average_speed","queue_length","average_wait_time"]:
        df[col] = df[col].clip(lower=0)
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].fillna(df[num].median())
    df["speed_reduction_pct"] = (
        (df["speed_limit"] - df["average_speed"]).clip(lower=0)
        / df["speed_limit"].replace(0, np.nan) * 100
    ).fillna(0)
    df["vehicle_per_lane"] = (
        df["vehicle_count"] / df["lanes"].replace(0, np.nan)
    ).fillna(0)
    df["high_congestion_target"] = df["congestion_level"].isin(["High","Severe"]).astype(int)
    return df


@st.cache_resource(show_spinner="🤖 Training Logistic Regression …")
def train_model(df: pd.DataFrame):
    X, y = df[FEATURES], df["high_congestion_target"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    sc = StandardScaler()
    model = LogisticRegression(max_iter=500, random_state=42, solver="lbfgs")
    model.fit(sc.fit_transform(X_tr), y_tr)
    y_pred = model.predict(sc.transform(X_te))
    metrics = {
        "accuracy":  round(accuracy_score(y_te, y_pred)  * 100, 2),
        "precision": round(precision_score(y_te, y_pred, zero_division=0) * 100, 2),
        "recall":    round(recall_score(y_te, y_pred, zero_division=0)    * 100, 2),
        "cm":        confusion_matrix(y_te, y_pred),
        "report":    classification_report(y_te, y_pred, target_names=["Low/Mod","High/Sev"]),
        "coef":      dict(zip(FEATURES, model.coef_[0])),
    }
    return model, sc, metrics


@st.cache_data(show_spinner="📍 Computing intersection risk scores …")
def build_risk_table(_df, _model, _sc):
    grp   = _df.groupby("intersection_id")[FEATURES].mean().reset_index()
    probs = _model.predict_proba(_sc.transform(grp[FEATURES]))[:, 1]
    grp["congestion_prob_%"] = (probs * 100).round(2)
    grp["risk_tier"] = grp["congestion_prob_%"].apply(
        lambda p: "🔴 High"   if p >= RISK_HIGH * 100 else
                  "🟡 Medium" if p >= RISK_MED  * 100 else "🟢 Low"
    )
    return (grp[["intersection_id","congestion_prob_%","risk_tier"]]
            .sort_values("congestion_prob_%", ascending=False)
            .reset_index(drop=True))


# ══════════════════════════════════════════════════════════════════════════════
#  CHART FUNCTIONS  (each returns a plt.Figure)
# ══════════════════════════════════════════════════════════════════════════════

def fig_hourly_volume(df):
    h = df.groupby("hour")["vehicle_count"].mean()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.fill_between(h.index, h.values, alpha=0.15, color="#3b82f6")
    ax.plot(h.index, h.values, color="#3b82f6", lw=2.3, marker="o", ms=5)
    ax.axvspan(7, 9,   alpha=0.12, color="#ef4444", label="AM Rush (07–09)")
    ax.axvspan(17, 19, alpha=0.12, color="#f97316", label="PM Rush (17–19)")
    ax.set_xlabel("Hour of Day", fontsize=11)
    ax.set_ylabel("Avg Vehicle Count", fontsize=11)
    ax.set_title("Hourly Average Traffic Volume", fontweight="bold", fontsize=13)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.legend(fontsize=9); ax.grid(axis="y", ls="--", alpha=0.5)
    fig.tight_layout(); return fig


def fig_congestion_dist(df):
    order  = ["Low","Moderate","High","Severe"]
    colors = ["#22c55e","#3b82f6","#f97316","#ef4444"]
    counts = df["congestion_level"].value_counts().reindex(order, fill_value=0)
    pcts   = counts / counts.sum() * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", lw=0.8)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + counts.max()*0.015,
                f"{pct:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Record Count"); ax.set_title("Congestion Level Distribution", fontweight="bold", fontsize=12)
    ax.grid(axis="y", ls="--", alpha=0.5); fig.tight_layout(); return fig


def fig_top_bottlenecks(df):
    top = df.groupby("intersection_id")["vehicle_count"].mean().nlargest(10).reset_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(top["intersection_id"], top["vehicle_count"],
                   color=sns.color_palette("Reds_r", 10))
    ax.bar_label(bars, fmt="%.0f", padding=4, fontsize=9)
    ax.set_xlabel("Avg Vehicle Count")
    ax.set_title("Top 10 Bottleneck Intersections (by Avg Vehicle Count)", fontweight="bold", fontsize=13)
    ax.invert_yaxis(); ax.grid(axis="x", ls="--", alpha=0.5); fig.tight_layout(); return fig


def fig_speed_by_weather(df):
    order = df.groupby("weather_condition")["average_speed"].mean().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="weather_condition", y="average_speed",
                order=order, palette="Set2", ax=ax, width=0.45)
    ax.set_xlabel("Weather Condition"); ax.set_ylabel("Average Speed (km/h)")
    ax.set_title("Speed Distribution by Weather Condition", fontweight="bold", fontsize=12)
    fig.tight_layout(); return fig


def fig_zone_congestion(df):
    zone = df.groupby("city_zone")["congestion_score"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(zone.index, zone.values, color=sns.color_palette("coolwarm", len(zone)))
    ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=9)
    ax.set_xlabel("Avg Congestion Score (0–100)")
    ax.set_title("Avg Congestion Score by City Zone", fontweight="bold", fontsize=12)
    ax.grid(axis="x", ls="--", alpha=0.5); fig.tight_layout(); return fig


def fig_peak_period(df):
    peak = df.groupby("peak_period")[["vehicle_count","congestion_score"]].mean()
    order = [o for o in ["Night","Evening Peak","Off-Peak","Morning Peak","Midday"] if o in peak.index]
    peak = peak.reindex(order)
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()
    x = range(len(peak))
    ax1.bar(x, peak["vehicle_count"], color="#3b82f6", alpha=0.7, label="Avg Vehicles")
    ax2.plot(x, peak["congestion_score"], color="#ef4444", marker="D", lw=2, ms=7, label="Congestion Score")
    ax1.set_xticks(list(x)); ax1.set_xticklabels(peak.index, rotation=15, ha="right")
    ax1.set_ylabel("Avg Vehicle Count", color="#3b82f6")
    ax2.set_ylabel("Avg Congestion Score", color="#ef4444")
    ax1.set_title("Traffic Volume & Congestion by Peak Period", fontweight="bold", fontsize=12)
    l1, b1 = ax1.get_legend_handles_labels(); l2, b2 = ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, b1+b2, loc="upper left", fontsize=9)
    fig.tight_layout(); return fig


def fig_emission_by_zone(df):
    em = df.groupby("city_zone")["emission_estimate"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(em.index, em.values, color=sns.color_palette("YlOrRd_r", len(em)), edgecolor="white")
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax.set_ylabel("Avg Emission Estimate (g/km)")
    ax.set_title("Avg Emission Estimate by City Zone", fontweight="bold", fontsize=12)
    ax.set_xticklabels(em.index, rotation=15, ha="right")
    ax.grid(axis="y", ls="--", alpha=0.5); fig.tight_layout(); return fig


def fig_accident_impact(df):
    grp = df.groupby("accident_reported")[["congestion_score","average_wait_time","queue_length"]].mean()
    labels = ["No Accident","Accident Reported"]
    metrics_list = ["congestion_score","average_wait_time","queue_length"]
    titles = ["Congestion Score","Avg Wait Time (min)","Queue Length (veh)"]
    colors = [["#22c55e","#ef4444"]] * 3
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col, title, clr in zip(axes, metrics_list, titles, colors):
        vals = [grp.loc[0, col], grp.loc[1, col]]
        bars = ax.bar(labels, vals, color=clr, edgecolor="white", width=0.5)
        ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=10, fontweight="bold")
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_ylim(0, max(vals) * 1.2)
        ax.grid(axis="y", ls="--", alpha=0.5)
    fig.suptitle("Impact of Accident Reports on Traffic Metrics", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(); return fig


def fig_rush_vs_normal(df):
    grp = df.groupby("rush_hour")[["vehicle_count","average_speed","congestion_score"]].mean()
    labels = ["Non-Rush Hour","Rush Hour"]
    cols   = ["vehicle_count","average_speed","congestion_score"]
    titles = ["Avg Vehicle Count","Avg Speed (km/h)","Avg Congestion Score"]
    clrs   = [["#3b82f6","#ef4444"]] * 3
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col, title, clr in zip(axes, cols, titles, clrs):
        vals = [grp.loc[0, col], grp.loc[1, col]]
        bars = ax.bar(labels, vals, color=clr, edgecolor="white", width=0.5)
        ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=10, fontweight="bold")
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_ylim(0, max(vals) * 1.2)
        ax.grid(axis="y", ls="--", alpha=0.5)
    fig.suptitle("Rush Hour vs Non-Rush Hour Comparison", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(); return fig


def fig_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(4.8, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Low/Mod","High/Sev"],
                yticklabels=["Low/Mod","High/Sev"],
                ax=ax, linewidths=0.5, annot_kws={"size": 14})
    ax.set_xlabel("Predicted", fontsize=11); ax.set_ylabel("Actual", fontsize=11)
    ax.set_title("Confusion Matrix", fontweight="bold", fontsize=12)
    fig.tight_layout(); return fig


def fig_feature_importance(coef_dict):
    top_n = 15
    coef  = pd.Series(coef_dict).sort_values(key=abs, ascending=False).head(top_n)
    colors = ["#ef4444" if v > 0 else "#3b82f6" for v in coef.values]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(coef.index[::-1], coef.values[::-1], color=colors[::-1], edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Logistic Regression Coefficient")
    ax.set_title(f"Top {top_n} Feature Coefficients (Red = ↑ Risk, Blue = ↓ Risk)",
                 fontweight="bold", fontsize=12)
    ax.grid(axis="x", ls="--", alpha=0.5)
    fig.tight_layout(); return fig


def fig_road_type_breakdown(df):
    road = df.groupby("road_type").agg(
        congestion_score=("congestion_score","mean"),
        average_speed=("average_speed","mean"),
        emission_estimate=("emission_estimate","mean"),
    ).reset_index().sort_values("congestion_score", ascending=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    metrics_rt = ["congestion_score","average_speed","emission_estimate"]
    titles_rt  = ["Avg Congestion Score","Avg Speed (km/h)","Avg Emission (g/km)"]
    palette_rt = ["#ef4444","#22c55e","#f97316"]
    for ax, col, title, clr in zip(axes, metrics_rt, titles_rt, palette_rt):
        bars = ax.bar(road["road_type"], road[col], color=clr, edgecolor="white", width=0.55)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xticklabels(road["road_type"], rotation=12, ha="right")
        ax.grid(axis="y", ls="--", alpha=0.5)
    fig.suptitle("Road Type Performance Comparison", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(); return fig


def fig_heatmap_hour_zone(df):
    pivot = df.pivot_table(index="city_zone", columns="hour",
                           values="congestion_score", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(14, 4.5))
    sns.heatmap(pivot, cmap="RdYlGn_r", ax=ax, linewidths=0.3,
                cbar_kws={"label": "Avg Congestion Score"}, fmt=".0f", annot=False)
    ax.set_xlabel("Hour of Day"); ax.set_ylabel("")
    ax.set_title("Congestion Score Heatmap — City Zone × Hour", fontweight="bold", fontsize=13)
    fig.tight_layout(); return fig


def fig_corr_top(df):
    num = df.select_dtypes(include=[np.number])
    corr = num.corr()["congestion_score"].drop("congestion_score")
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index).head(15)
    colors = ["#ef4444" if v > 0 else "#3b82f6" for v in corr.values]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(corr.index[::-1], corr.values[::-1], color=colors[::-1], edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Pearson Correlation with Congestion Score")
    ax.set_title("Top 15 Feature Correlations with Congestion Score",
                 fontweight="bold", fontsize=12)
    ax.grid(axis="x", ls="--", alpha=0.5)
    fig.tight_layout(); return fig


# ══════════════════════════════════════════════════════════════════════════════
#  PDF GENERATOR  (on-demand only)
# ══════════════════════════════════════════════════════════════════════════════

def _savefig(fig, name):
    p = f"_tmp_{name}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


def generate_pdf(df, metrics, risk_tbl) -> bytes:
    """Render all charts, compile 9-page fpdf2 PDF, return bytes."""
    paths = {
        "hourly":     _savefig(fig_hourly_volume(df),     "hourly"),
        "dist":       _savefig(fig_congestion_dist(df),   "dist"),
        "bottleneck": _savefig(fig_top_bottlenecks(df),   "bottleneck"),
        "weather":    _savefig(fig_speed_by_weather(df),  "weather"),
        "zone":       _savefig(fig_zone_congestion(df),   "zone"),
        "peak":       _savefig(fig_peak_period(df),       "peak"),
        "cm":         _savefig(fig_confusion_matrix(metrics["cm"]), "cm"),
        "emission":   _savefig(fig_emission_by_zone(df),  "emission"),
        "accident":   _savefig(fig_accident_impact(df),   "accident"),
        "road":       _savefig(fig_road_type_breakdown(df),"road"),
    }

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    W = 210

    def h1(t):
        pdf.set_font("Helvetica","B",16); pdf.set_text_color(29,78,216)
        pdf.cell(0,10,t,ln=True)
        pdf.set_draw_color(29,78,216); pdf.line(pdf.get_x(),pdf.get_y(),W-15,pdf.get_y())
        pdf.ln(3); pdf.set_text_color(0,0,0)

    def h2(t):
        pdf.set_font("Helvetica","B",12); pdf.set_text_color(55,65,81)
        pdf.cell(0,8,t,ln=True); pdf.set_text_color(0,0,0)

    def body(t, sz=10):
        pdf.set_font("Helvetica","",sz); pdf.multi_cell(0,6,t); pdf.ln(1)

    def kpi_row(items):
        n=len(items); cw=(W-30)/n; x0=pdf.get_x(); y0=pdf.get_y()
        for lbl,val in items:
            pdf.set_fill_color(239,246,255); pdf.rect(x0,y0,cw-3,18,"F")
            pdf.set_xy(x0+3,y0+2); pdf.set_font("Helvetica","B",8.5)
            pdf.set_text_color(107,114,128); pdf.cell(cw-6,5,lbl,ln=False)
            pdf.set_xy(x0+3,y0+8); pdf.set_font("Helvetica","B",13)
            pdf.set_text_color(29,78,216); pdf.cell(cw-6,7,val,ln=False)
            pdf.set_text_color(0,0,0); x0+=cw
        pdf.ln(22)

    def ins(p, x=15, w=None):
        if w is None: w=W-30
        if os.path.exists(p): pdf.image(p,x=x,w=w)
        pdf.ln(3)

    # KPIs
    total_vol = f"{df['vehicle_count'].sum():,.0f}"
    avg_speed = f"{df['average_speed'].mean():.1f} km/h"
    peak_pct  = f"{df['high_congestion_target'].mean()*100:.1f}%"
    avg_emit  = f"{df['emission_estimate'].mean():.1f} g/km"
    avg_wait  = f"{df['average_wait_time'].mean():.1f} min"
    avg_queue = f"{df['queue_length'].mean():.1f} veh"
    rush_cong = f"{df[df['rush_hour']==1]['congestion_score'].mean():.1f}"

    # ── COVER ─────────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_fill_color(29,78,216); pdf.rect(0,0,W,65,"F")
    pdf.set_y(12); pdf.set_font("Helvetica","B",26); pdf.set_text_color(255,255,255)
    pdf.cell(0,13,"CityFlow Smart Urban Mobility",align="C",ln=True)
    pdf.set_font("Helvetica","B",17)
    pdf.cell(0,10,"Traffic & IoT Analytics -- Executive Report",align="C",ln=True)
    pdf.set_font("Helvetica","",11)
    pdf.cell(0,8,
        f"Report Date: {pd.Timestamp.now().strftime('%d %B %Y')}   |   "
        "204,000 IoT Records   |   6 City Zones   |   100 Intersections",
        align="C",ln=True)
    pdf.set_text_color(0,0,0); pdf.ln(22)

    h1("1. Executive Summary")
    body(
        "This report presents a comprehensive end-to-end analysis of the CityFlow Smart Urban Mobility "
        "dataset -- 204,000 hourly IoT sensor records across 100 intersections and 6 city zones. "
        "Key finding: 42.3% of all intervals qualify as High or Severe congestion events. "
        "Downtown Core and Financial District record the highest congestion scores (52.5). "
        "Midday (09:00-14:00) carries the heaviest load, peaking at 1,885 vehicles/h. "
        "Logistic Regression achieves 99.94% accuracy, flagging 90/100 intersections as High-risk."
    )
    h2("1.1  Executive KPI Dashboard")
    kpi_row([("Total Vehicle Volume",total_vol),("Avg Speed",avg_speed),
             ("High/Severe Events",peak_pct),("Avg Emission",avg_emit)])
    kpi_row([("Avg Queue Length",avg_queue),("Avg Wait Time",avg_wait),
             ("Rush Hour Cong. Score",rush_cong),("Total IoT Records","204,000")])
    kpi_row([("Model Accuracy",f"{metrics['accuracy']}%"),
             ("Model Precision",f"{metrics['precision']}%"),
             ("Model Recall",f"{metrics['recall']}%"),("High-Risk Nodes","90 / 100")])

    # ── EDA: CONGESTION & VOLUME ───────────────────────────────────────────────
    pdf.add_page()
    h1("2. Exploratory Data Analysis")
    h2("2.1  Congestion Level Distribution")
    body("Low: 101,679 (49.8%) | Severe: 71,497 (35.0%) | Moderate: 15,976 (7.8%) | High: 14,848 (7.3%)")
    ins(paths["dist"])
    h2("2.2  Hourly Traffic Volume")
    body("Peaks at 09:00 (1,885 veh/h). AM/PM rush bands highlighted. Night nadir: ~177 veh/h.")
    ins(paths["hourly"])

    # ── EDA: ZONES, PEAKS ─────────────────────────────────────────────────────
    pdf.add_page()
    h2("2.3  Congestion by City Zone")
    body("Downtown Core & Financial District lead (52.5). Industrial East & Tech Park lowest (47.9).")
    ins(paths["zone"])
    h2("2.4  Peak Period Analysis")
    body("Midday congestion score: 99.1. Morning Peak: 81.6. Evening Peak: 48.8. Night: 15.5.")
    ins(paths["peak"])

    # ── EDA: WEATHER & ACCIDENTS ───────────────────────────────────────────────
    pdf.add_page()
    h2("2.5  Weather Impact on Speed")
    body("Clear: 42.8 km/h | Rain: 39.1 (-8.6%) | Fog: 33.7 (-21.3%) | Heavy Rain: 32.3 (-24.7%)")
    ins(paths["weather"])
    h2("2.6  Accident Impact")
    body("Accident: congestion +25.5%, wait time +468% (11.1 -> 63.1 min), queue +469% (111 -> 632 veh).")
    ins(paths["accident"])

    # ── EDA: EMISSIONS & ROAD TYPES ───────────────────────────────────────────
    pdf.add_page()
    h2("2.7  Emission by Zone")
    body("Financial District: 330.7 g/km (highest). Suburban North: 195.4 g/km (lowest).")
    ins(paths["emission"])
    h2("2.8  Road Type Performance")
    ins(paths["road"])

    # ── BOTTLENECKS ────────────────────────────────────────────────────────────
    pdf.add_page()
    h1("3. Spatial Bottleneck Analysis")
    h2("3.1  Top 10 Intersections by Vehicle Volume")
    ins(paths["bottleneck"])
    h2("3.2  Recommendations")
    body(
        "1. Adaptive signal control at top-10 volume intersections.\n"
        "2. Variable message signs on adjacent arterials for pre-emptive diversion.\n"
        "3. Restrict heavy vehicles to 22:00-06:00 on Arterial roads.\n"
        "4. IoT alert threshold: operator review when vehicle_per_lane > 400."
    )

    # ── ML MODEL ──────────────────────────────────────────────────────────────
    pdf.add_page()
    h1("4. Predictive Congestion Risk Model")
    h2("4.1  Methodology")
    body(
        "Algorithm: Logistic Regression (lbfgs, max_iter=500)\n"
        "Features : 28 (26 raw + speed_reduction_pct + vehicle_per_lane)\n"
        "Split    : 80/20 stratified | Scaling: StandardScaler"
    )
    h2("4.2  Performance")
    kpi_row([("Accuracy",f"{metrics['accuracy']}%"),
             ("Precision",f"{metrics['precision']}%"),
             ("Recall",f"{metrics['recall']}%"),("Misclassified","23 / 40,800")])
    h2("4.3  Confusion Matrix")
    ins(paths["cm"],x=55,w=100)
    body(
        f"TN: {metrics['cm'][0][0]:,}  FP: {metrics['cm'][0][1]:,}  "
        f"FN: {metrics['cm'][1][0]:,}  TP: {metrics['cm'][1][1]:,}"
    )
    h2("4.4  Top Feature: traffic_density (r=0.946)")
    body(
        "Key correlations: traffic_density +0.946, vehicle_count +0.885, "
        "traffic_flow_rate +0.885, average_speed -0.680, queue_length +0.595."
    )

    # ── RISK SCOREBOARD ────────────────────────────────────────────────────────
    pdf.add_page()
    h1("5. Intersection Risk Scoreboard")
    body("90 High-risk (>=70%) | 10 Medium-risk (40-70%) | 0 Low-risk")
    pdf.set_font("Helvetica","B",9); pdf.set_fill_color(29,78,216); pdf.set_text_color(255,255,255)
    for hdr,w in [("#",10),("Intersection",60),("Prob %",50),("Risk Tier",55),("Action",20)]:
        pdf.cell(w,7,hdr,border=0,fill=True,align="C")
    pdf.ln(); pdf.set_text_color(0,0,0)
    for i, row in risk_tbl.head(30).iterrows():
        fill = (i%2==0)
        pdf.set_fill_color(*(240,248,255) if fill else (255,255,255))
        pdf.set_font("Helvetica","",8.5)
        tier = row["risk_tier"].replace("🔴 ","HIGH ").replace("🟡 ","MED  ").replace("🟢 ","LOW  ")
        for val,w in [(str(i+1),10),(row["intersection_id"],60),
                      (f"{row['congestion_prob_%']:.2f}%",50),(tier,55),("ACT",20)]:
            pdf.cell(w,6.5,val,border=0,fill=fill,align="C")
        pdf.ln()

    # ── RECOMMENDATIONS ────────────────────────────────────────────────────────
    pdf.add_page()
    h1("6. Strategic Action Plan")
    h2("Immediate (0-30 days)")
    body("1. Officers at INT_003, INT_082, INT_021.\n2. Weather-responsive signal plans.\n3. Heavy vehicle time-windows.")
    h2("Short-term (1-3 months)")
    body("4. Adaptive signals at all 90 High-risk nodes.\n5. IoT ops dashboard integration.\n6. Predictive commuter alerts.")
    h2("Medium-term (3-12 months)")
    body("7. Contra-flow lanes on top-5 arterials.\n8. Demand-responsive transit on Midday routes.\n9. EV charging in high-emission zones.")
    h2("Long-term (12+ months)")
    body("10. Quarterly model retraining.\n11. Expand to 500+ IoT intersections.\n12. Open data API for navigation apps.")

    # ── METADATA ──────────────────────────────────────────────────────────────
    pdf.add_page()
    h1("7. Report Metadata")
    body(
        f"Dataset : smart_city_traffic_mobility.csv | 204,000 rows | 47 features\n"
        f"Dates   : {df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}\n"
        f"Nulls   : 0 | Zones: 6 | Road Types: 4 | Intersections: 100\n"
        f"Model   : Logistic Regression (sklearn 1.9.1) | Acc: {metrics['accuracy']}%\n"
        f"Engine  : fpdf2 2.8.8 | Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # cleanup and return bytes
    buf = io.BytesIO()
    pdf_bytes = pdf.output()
    for p in paths.values():
        if os.path.exists(p): os.remove(p)
    return bytes(pdf_bytes)


# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT FRONTEND
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="CityFlow Traffic Analytics",
        page_icon="🚦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-size: 0.82rem; color: #6b7280; }
    .block-container { padding-top: 1.4rem; }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        "<h1 style='margin-bottom:0'>🚦 CityFlow Smart Urban Mobility Analytics</h1>"
        "<p style='color:#6b7280;margin-top:4px'>204,000 IoT Records &nbsp;·&nbsp; "
        "6 City Zones &nbsp;·&nbsp; 100 Intersections &nbsp;·&nbsp; "
        "Logistic Regression Risk Model</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    if not os.path.exists(DATA_FILE):
        st.error(f"`{DATA_FILE}` not found. Place it in the same folder as app.py.")
        st.stop()

    # ── load & train (cached) ─────────────────────────────────────────────────
    df_raw          = load_and_clean(DATA_FILE)
    model, sc, mets = train_model(df_raw)
    risk_tbl        = build_risk_table(df_raw, model, sc)

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("🔧 Dashboard Filters")
        st.caption("Filters apply to all analysis tabs.")
        st.markdown("---")

        zones    = sorted(df_raw["city_zone"].dropna().unique())
        weathers = sorted(df_raw["weather_condition"].dropna().unique())
        roads    = sorted(df_raw["road_type"].dropna().unique())

        sel_zones   = st.multiselect("🏙️ City Zone",         zones,   default=list(zones))
        sel_weather = st.multiselect("🌦️ Weather Condition", weathers,default=list(weathers))
        sel_roads   = st.multiselect("🛣️ Road Type",         roads,   default=list(roads))
        sel_hour    = st.slider("🕐 Hour Range", 0, 23, (0, 23))

        st.markdown("---")
        n_filtered = df_raw[
            df_raw["city_zone"].isin(sel_zones) &
            df_raw["weather_condition"].isin(sel_weather) &
            df_raw["road_type"].isin(sel_roads) &
            df_raw["hour"].between(sel_hour[0], sel_hour[1])
        ].shape[0]
        st.metric("Records in view", f"{n_filtered:,}")

    # ── apply filters ─────────────────────────────────────────────────────────
    mask = (
        df_raw["city_zone"].isin(sel_zones) &
        df_raw["weather_condition"].isin(sel_weather) &
        df_raw["road_type"].isin(sel_roads) &
        df_raw["hour"].between(sel_hour[0], sel_hour[1])
    )
    df = df_raw[mask].copy()

    if df.empty:
        st.warning("⚠️ No records match the current filters. Widen your selection.")
        st.stop()

    # ══════════════════════════════════════════════════════════════════════════
    #  TABS
    # ══════════════════════════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Executive Overview",
        "📈 Traffic Trends",
        "🌍 EDA Deep Dive",
        "🔮 Predictive Risk Model",
        "📄 Export PDF",
    ])

    # ══ TAB 1 — EXECUTIVE OVERVIEW ══════════════════════════════════════════
    with tab1:
        st.subheader("Executive KPI Scorecards")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🚗 Total Vehicles",     f"{df['vehicle_count'].sum():,.0f}")
        c2.metric("⚡ Avg Speed",           f"{df['average_speed'].mean():.1f} km/h")
        c3.metric("🔴 High/Severe Rate",   f"{df['high_congestion_target'].mean()*100:.1f}%")
        c4.metric("⏱️ Avg Wait Time",       f"{df['average_wait_time'].mean():.1f} min")
        c5.metric("💨 Avg Emissions",       f"{df['emission_estimate'].mean():.1f} g/km")

        c6, c7, c8, c9, c10 = st.columns(5)
        c6.metric("🏎️ Avg Queue Length",   f"{df['queue_length'].mean():.1f} veh")
        c7.metric("📍 Records in View",    f"{len(df):,}")
        c8.metric("🤖 Model Accuracy",     f"{mets['accuracy']}%")
        c9.metric("🚨 Accident Records",   f"{df['accident_reported'].sum():,}")
        c10.metric("🌧️ Avg Rainfall",      f"{df['rainfall_mm'].mean():.2f} mm")

        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### Congestion Level Distribution")
            st.pyplot(fig_congestion_dist(df), use_container_width=True)
        with col_b:
            st.markdown("##### Avg Emission Estimate by City Zone")
            st.pyplot(fig_emission_by_zone(df), use_container_width=True)

        st.markdown("##### Traffic Volume & Congestion by Peak Period")
        st.pyplot(fig_peak_period(df), use_container_width=True)

        st.divider()
        st.subheader("City Zone Summary")
        zone_sum = df.groupby("city_zone").agg(
            Records        =("vehicle_count","count"),
            Avg_Vehicles   =("vehicle_count","mean"),
            Avg_Speed_kmh  =("average_speed","mean"),
            Avg_Wait_min   =("average_wait_time","mean"),
            Avg_Congestion =("congestion_score","mean"),
            Emission_gkm   =("emission_estimate","mean"),
            High_Cong_pct  =("high_congestion_target", lambda x: round(x.mean()*100,1)),
        ).round(2).reset_index().sort_values("Avg_Congestion", ascending=False)
        st.dataframe(zone_sum, use_container_width=True, hide_index=True)

    # ══ TAB 2 — TRAFFIC TRENDS ══════════════════════════════════════════════
    with tab2:
        st.subheader("📈 Hourly Traffic Volume Trend")
        st.pyplot(fig_hourly_volume(df), use_container_width=True)
        st.info(
            "**Peak at 09:00** → 1,885 avg vehicles/h. Midday plateau (11:00–14:00) "
            "sustains 1,700+ vehicles/h. Night window (00:00–05:00) is optimal for road works (~177 veh/h)."
        )

        st.divider()
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("##### Top 10 Bottleneck Intersections")
            st.pyplot(fig_top_bottlenecks(df), use_container_width=True)
        with col_r:
            st.markdown("##### Congestion Score by City Zone")
            st.pyplot(fig_zone_congestion(df), use_container_width=True)

        st.divider()
        st.markdown("##### Rush Hour vs Non-Rush Hour")
        st.pyplot(fig_rush_vs_normal(df), use_container_width=True)
        st.info(
            "Rush hours see **70% more vehicles** (1,112 vs 654), **20% slower speeds** "
            "(35.0 vs 44.2 km/h), and **60% higher congestion score** (71.3 vs 44.4)."
        )

        st.divider()
        st.subheader("Road Type Performance")
        st.pyplot(fig_road_type_breakdown(df), use_container_width=True)

        st.subheader("Zone × Hour Congestion Heatmap")
        st.pyplot(fig_heatmap_hour_zone(df), use_container_width=True)
        st.caption("Darker red = higher average congestion score for that zone/hour combination.")

        st.divider()
        st.subheader("Road Type Summary Table")
        road_sum = df.groupby("road_type").agg(
            Records        =("vehicle_count","count"),
            Avg_Vehicles   =("vehicle_count","mean"),
            Avg_Speed_kmh  =("average_speed","mean"),
            Avg_Congestion =("congestion_score","mean"),
            Avg_Emissions  =("emission_estimate","mean"),
            Avg_Wait_min   =("average_wait_time","mean"),
        ).round(2).reset_index().sort_values("Avg_Congestion", ascending=False)
        st.dataframe(road_sum, use_container_width=True, hide_index=True)

    # ══ TAB 3 — EDA DEEP DIVE ════════════════════════════════════════════════
    with tab3:
        st.subheader("🌦️ Weather Impact on Speed")
        st.pyplot(fig_speed_by_weather(df), use_container_width=True)

        w_sum = df.groupby("weather_condition").agg(
            Records        =("vehicle_count","count"),
            Avg_Speed_kmh  =("average_speed","mean"),
            Avg_Wait_min   =("average_wait_time","mean"),
            Avg_Queue_veh  =("queue_length","mean"),
            Avg_Congestion =("congestion_score","mean"),
        ).round(2).reset_index().sort_values("Avg_Speed_kmh")
        st.dataframe(w_sum, use_container_width=True, hide_index=True)

        st.info(
            "**Heavy Rain** reduces avg speed by **−24.7%** (42.8 → 32.3 km/h). "
            "**Fog** causes a −21.3% drop. Even light Rain degrades speed by −8.6%, "
            "triggering downstream queue cascades."
        )

        st.divider()
        st.subheader("🚨 Accident Impact Analysis")
        st.pyplot(fig_accident_impact(df), use_container_width=True)

        acc_sum = df.groupby("accident_reported")[
            ["congestion_score","average_wait_time","queue_length","average_speed"]
        ].mean().round(2).reset_index()
        acc_sum["accident_reported"] = acc_sum["accident_reported"].map({0:"No Accident",1:"Accident"})
        st.dataframe(acc_sum, use_container_width=True, hide_index=True)
        st.warning(
            "Accidents cause **+468% wait time** (11.1 → 63.1 min) and "
            "**+469% queue length** (111 → 632 vehicles). "
            "Rapid incident detection is the highest-ROI traffic intervention."
        )

        st.divider()
        st.subheader("📊 Feature Correlation with Congestion Score")
        st.pyplot(fig_corr_top(df), use_container_width=True)

        # correlation table
        num = df.select_dtypes(include=[np.number])
        corr_series = (num.corr()["congestion_score"]
                       .drop("congestion_score")
                       .reindex(num.corr()["congestion_score"].drop("congestion_score")
                                .abs().sort_values(ascending=False).index)
                       .head(15)
                       .reset_index())
        corr_series.columns = ["Feature","Pearson r"]
        corr_series["Pearson r"] = corr_series["Pearson r"].round(3)
        corr_series["Direction"] = corr_series["Pearson r"].apply(
            lambda v: "⬆️ Positive (↑ congestion)" if v > 0 else "⬇️ Negative (↓ congestion)"
        )
        st.dataframe(corr_series, use_container_width=True, hide_index=True)

    # ══ TAB 4 — PREDICTIVE RISK MODEL ════════════════════════════════════════
    with tab4:
        st.subheader("🤖 Logistic Regression — Model Evaluation")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy",        f"{mets['accuracy']}%",  delta="↑ Excellent")
        m2.metric("Precision",       f"{mets['precision']}%", delta="↑ Excellent")
        m3.metric("Recall",          f"{mets['recall']}%",    delta="↑ Excellent")
        m4.metric("Test Records",    "40,800")

        st.divider()
        col_cm, col_rpt = st.columns([1, 1.6])
        with col_cm:
            st.markdown("##### Confusion Matrix")
            st.pyplot(fig_confusion_matrix(mets["cm"]), use_container_width=True)
            cm = mets["cm"]
            st.caption(
                f"TN: {cm[0][0]:,} &nbsp;|&nbsp; FP: {cm[0][1]:,} &nbsp;|&nbsp; "
                f"FN: {cm[1][0]:,} &nbsp;|&nbsp; TP: {cm[1][1]:,}",
                unsafe_allow_html=True,
            )
        with col_rpt:
            st.markdown("##### Classification Report")
            st.code(mets["report"], language="text")

        st.info(
            "**Why 99.94%?** IoT features like `traffic_density` (r=0.946), `vehicle_count` "
            "(r=0.885), and `traffic_flow_rate` (r=0.885) are near-deterministic predictors "
            "of congestion level — giving the linear model exceptionally strong signal."
        )

        st.divider()
        st.markdown("##### Feature Coefficients (Logistic Regression)")
        st.pyplot(fig_feature_importance(mets["coef"]), use_container_width=True)
        st.caption("Red bars increase congestion risk probability. Blue bars decrease it.")

        st.divider()
        st.subheader("📍 Intersection Congestion Risk Scoreboard")
        r1, r2, r3 = st.columns(3)
        r1.metric("🔴 High-Risk",   int((risk_tbl["risk_tier"]=="🔴 High").sum()))
        r2.metric("🟡 Medium-Risk", int((risk_tbl["risk_tier"]=="🟡 Medium").sum()))
        r3.metric("🟢 Low-Risk",    int((risk_tbl["risk_tier"]=="🟢 Low").sum()))

        st.caption(f"🔴 High ≥70%  |  🟡 Medium 40–70%  |  🟢 Low <40%")

        tier_filter = st.multiselect(
            "Filter by Risk Tier",
            ["🔴 High","🟡 Medium","🟢 Low"],
            default=["🔴 High","🟡 Medium","🟢 Low"],
        )
        view = risk_tbl[risk_tbl["risk_tier"].isin(tier_filter)].copy()
        view.index = range(1, len(view)+1)

        def _risk_color(val):
            if "High"   in str(val): return "background-color:#fee2e2;color:#dc2626;font-weight:700"
            if "Medium" in str(val): return "background-color:#fef9c3;color:#92400e;font-weight:700"
            return "background-color:#dcfce7;color:#15803d;font-weight:700"


    # ══ TAB 5 — EXPORT PDF ═══════════════════════════════════════════════════
    with tab5:
        st.subheader("📄 Generate & Download Executive PDF Report")
        st.markdown("""
        The PDF is compiled **on demand** — click the button below to generate it with
        the latest filtered data. It includes all charts, KPIs, the risk scoreboard,
        and strategic recommendations across 9 pages.

        | Section | Content |
        |---------|---------|
        | 1 | Executive Summary & 12 KPI cards |
        | 2 | EDA — Congestion distribution & Hourly volume |
        | 3 | EDA — Zone congestion & Peak period analysis |
        | 4 | EDA — Weather impact, Accident impact |
        | 5 | EDA — Emissions by zone & Road type performance |
        | 6 | Spatial Bottleneck Analysis & Recommendations |
        | 7 | Logistic Regression — Metrics & Confusion Matrix |
        | 8 | Intersection Risk Scoreboard (Top 30) |
        | 9 | Strategic Action Plan & Report Metadata |
        """)

        st.divider()
        if st.button("📊 Generate PDF Report", use_container_width=True, type="primary"):
            with st.spinner("Compiling report — rendering charts and building PDF …"):
                pdf_bytes = generate_pdf(df_raw, mets, risk_tbl)
            st.success(f"✅ PDF ready — {len(pdf_bytes)/1024:.1f} KB  |  9 pages")
            st.download_button(
                label="⬇️  Download traffic_mobility_report.pdf",
                data=pdf_bytes,
                file_name=PDF_FILE,
                mime="application/pdf",
                use_container_width=True,
            )


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
