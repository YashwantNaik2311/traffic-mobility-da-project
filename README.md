# 🚦 CityFlow Smart Urban Mobility & Traffic Analytics

> **End-to-end Python data analytics project** — IoT traffic data → EDA → ML prediction → interactive dashboard → automated PDF report.

---

## 📋 Project Overview

This project performs a full-stack analysis of the **CityFlow Smart Urban Mobility dataset** — 204,000 hourly IoT sensor readings across **100 intersections**, **6 city zones**, and **4 road types**. It delivers:

- 📊 **Interactive Streamlit dashboard** with sidebar filters and 4 analytical tabs
- 🤖 **Logistic Regression model** predicting high-congestion risk (99.94% accuracy)
- 📄 **9-page executive PDF report** auto-generated with `fpdf2`
- 🔍 **Intersection risk scoreboard** ranking all 100 nodes into High / Medium / Low tiers

---

## 📁 Project Structure

```
DA Internship Project/
│
├── app.py                          # Full-stack Streamlit app (backend + frontend)
├── requirements.txt                # Python dependencies
├── smart_city_traffic_mobility.csv # Source dataset (204,000 rows × 47 features)
├── traffic_mobility_report.pdf     # Auto-generated executive PDF report
└── README.md                       # This file
```

---

## 📦 Dataset Schema

| Feature | Type | Description |
|---------|------|-------------|
| `intersection_id` | str | Unique intersection identifier (INT_001 … INT_100) |
| `city_zone` | str | Downtown Core, Financial District, Suburban North, etc. |
| `road_type` | str | Highway, Arterial, Collector, Local Street |
| `vehicle_count` | int | Total vehicles recorded in the hour |
| `average_speed` | float | Mean speed (km/h) |
| `traffic_density` | float | Vehicles per km (strongest congestion predictor, r=0.946) |
| `congestion_score` | float | Composite score 0–100 |
| `congestion_level` | str | Low / Moderate / High / Severe |
| `weather_condition` | str | Clear / Rain / Heavy Rain / Fog |
| `rush_hour` | int | 1 = AM or PM rush window |
| `emission_estimate` | float | Estimated emissions (g/km) |
| `timestamp` | datetime | Hourly ISO timestamp |
| *(+ 35 more features)* | | |

**Engineered features added during preprocessing:**

| Feature | Formula |
|---------|---------|
| `speed_reduction_pct` | `(speed_limit − avg_speed) / speed_limit × 100` |
| `vehicle_per_lane` | `vehicle_count / lanes` |
| `high_congestion_target` | `1` if `congestion_level` ∈ {High, Severe} else `0` |

---

## 📊 Key Analysis Findings

### Congestion Distribution
| Level | Count | Share |
|-------|-------|-------|
| Low | 101,679 | 49.8% |
| Severe | 71,497 | 35.0% |
| Moderate | 15,976 | 7.8% |
| High | 14,848 | 7.3% |

> The dataset is **bimodal** — skewed toward Low and Severe with few Moderate records, suggesting rapid escalation at critical congestion thresholds.

### Hourly Traffic Peaks
| Hour | Avg Vehicles/h | Peak Period |
|------|---------------|-------------|
| 09:00 | 1,885 | Morning Rush |
| 12:00 | 1,761 | Midday |
| 11:00 | 1,705 | Midday |
| 00:00–05:00 | ~177 | Night (lowest) |

### Weather Impact on Speed
| Condition | Avg Speed | vs Clear |
|-----------|-----------|---------|
| Clear | 42.8 km/h | baseline |
| Rain | 39.1 km/h | −8.6% |
| Fog | 33.7 km/h | −21.3% |
| Heavy Rain | 32.3 km/h | **−24.7%** |

### Accident Impact
| Metric | No Accident | With Accident | Change |
|--------|-------------|---------------|--------|
| Congestion Score | 49.8 | 62.5 | +25.5% |
| Avg Wait Time | 11.1 min | 63.1 min | **+468%** |
| Queue Length | 111 veh | 632 veh | **+469%** |

### City Zone Rankings (by Congestion Score)
| Zone | Congestion Score | Emission (g/km) |
|------|-----------------|-----------------|
| Downtown Core | 52.5 | 310.0 |
| Financial District | 52.5 | **330.7** |
| Suburban North | 47.9 | 195.4 |
| Residential West | 47.9 | 200.9 |
| Tech Park | 47.9 | 221.0 |
| Industrial East | 47.9 | 208.1 |

---

## 🤖 Predictive Model

| Attribute | Value |
|-----------|-------|
| Algorithm | Logistic Regression (sklearn, lbfgs) |
| Features | 28 (26 raw + 2 engineered) |
| Train / Test | 163,200 / 40,800 (80/20 stratified) |
| **Accuracy** | **99.94%** |
| **Precision** | **99.93%** |
| **Recall** | **99.94%** |
| Misclassifications | 23 / 40,800 |

**Top predictive features** (Pearson r with `congestion_score`):
1. `traffic_density` — r = **+0.946**
2. `vehicle_count` — r = +0.885
3. `traffic_flow_rate` — r = +0.885
4. `average_speed` — r = **−0.680** (inverse)
5. `queue_length` — r = +0.595

**Intersection Risk Results:**
- 🔴 **High Risk** (≥70%): **90 intersections**
- 🟡 **Medium Risk** (40–70%): **10 intersections**
- 🟢 **Low Risk** (<40%): **0 intersections**

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

> The PDF report is **auto-generated on first load** and available for download in **Tab 4**.

---

## 🖥️ Dashboard Tabs

| Tab | Content |
|-----|---------|
| 📊 **Executive Overview** | 10 KPI metric cards, congestion distribution, emissions by zone, peak period chart, zone summary table |
| 📈 **Traffic Trends & Bottlenecks** | Hourly volume trend, top-10 bottleneck intersections, speed by weather, congestion by zone, road-type & weather summary tables |
| 🔮 **Predictive Risk Model** | Model metrics, confusion matrix, classification report, colour-coded risk scoreboard for all 100 intersections |
| 📄 **PDF Export** | One-click download of the full 9-page executive report |

### Sidebar Filters (apply to Tabs 1 & 2)
- 🏙️ **City Zone** — multi-select (6 zones)
- 🌦️ **Weather Condition** — multi-select (Clear / Rain / Heavy Rain / Fog)
- 🕐 **Hour Range** — slider (0–23)
- 🛣️ **Road Type** — multi-select (Highway / Arterial / Collector / Local Street)

---

## 📄 PDF Report Contents

The auto-generated `traffic_mobility_report.pdf` (9 pages) includes:

1. **Executive Summary** — KPI dashboard with 12 live metrics
2. **EDA: Congestion & Volume** — distribution chart + hourly trend chart
3. **EDA: Zones & Peak Periods** — zone congestion bar + dual-axis peak chart
4. **EDA: Weather, Accidents & Emissions** — box plot + accident impact table + emission chart
5. **Spatial Bottleneck Analysis** — top-10 intersection bar chart + recommendations
6. **Predictive Model** — methodology, metrics, confusion matrix, feature importance
7. **Risk Scoreboard** — top-30 intersections with probability scores and tier labels
8. **Strategic Action Plan** — 16 recommendations across 4 time horizons
9. **Report Metadata** — data quality, model details, generation timestamp

---

## ⚙️ Requirements

```
streamlit>=1.35.0
pandas>=2.2.0
numpy>=1.26.0
scikit-learn>=1.5.0
matplotlib>=3.9.0
seaborn>=0.13.0
fpdf2>=2.7.9
```

> Tested on Python 3.14 | Windows 11

---

## 🏗️ Architecture

```
app.py
├── BACKEND
│   ├── load_and_clean()       → parse timestamps, clip outliers, engineer 3 new features
│   ├── train_model()          → Logistic Regression, 80/20 split, StandardScaler
│   ├── build_risk_table()     → per-intersection congestion probability + risk tier
│   ├── chart_*()              → 8 matplotlib/seaborn chart functions
│   └── generate_pdf()         → 9-page fpdf2 PDF with charts + real data KPIs
│
└── FRONTEND (Streamlit)
    ├── Sidebar  → City Zone · Weather · Hour Range · Road Type filters
    ├── Tab 1    → 10 KPI metric cards + 3 charts + zone summary table
    ├── Tab 2    → Hourly trend + bottleneck bar + weather box + zone bar + 2 summary tables
    ├── Tab 3    → Model metrics + confusion matrix + classification report + risk scoreboard
    └── Tab 4    → PDF download button + regenerate option
```

---

## 📌 Strategic Recommendations Summary

| Horizon | Priority Actions |
|---------|-----------------|
| **Immediate** (0–30 days) | Emergency officers at top-3 risk intersections; weather-responsive signal plans |
| **Short-term** (1–3 months) | Adaptive signal controllers at all 90 High-risk nodes; real-time IoT ops dashboard |
| **Medium-term** (3–12 months) | Contra-flow lanes on top-5 arterials; demand-responsive transit; EV charging expansion |
| **Long-term** (12+ months) | Quarterly model retraining; expand IoT to 500+ intersections; open data API |

---

## 📝 Author Notes

- **Zero missing values** in the dataset after cleaning (confirmed: nulls = 0)
- Model accuracy is exceptionally high because IoT features like `traffic_density` and `traffic_flow_rate` are near-deterministic predictors — a real-world scenario where sensor quality is high
- All charts use `matplotlib.use("Agg")` for non-interactive server rendering, ensuring compatibility in headless Streamlit environments

---

*CityFlow Smart Urban Mobility Analytics | DA Internship Project | September 2026*
