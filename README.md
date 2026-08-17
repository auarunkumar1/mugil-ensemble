# MUGIL v19.3 — Complete Pipeline Technical Reference

> A multi-model ensemble weather forecasting system for South India, producing
> interpolated meteorological maps from 5 global NWP/AI models plus ECMWF
> high-resolution GRIB data.

---

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [Station Network](#2-station-network)
3. [Data Acquisition](#3-data-acquisition)
4. [Ensemble Weighting](#4-ensemble-weighting)
5. [Derived Variables & Indices](#5-derived-variables--indices)
6. [Spatial Interpolation & Terrain Correction](#6-spatial-interpolation--terrain-correction)
7. [ECMWF GRIB Automation](#7-ecmwf-grib-automation)
8. [Map Rendering & Output Products](#8-map-rendering--output-products)
9. [Convergence Calculation](#9-convergence-calculation)
10. [Known Approximations](#10-known-approximations)

---

## 1. Architecture Overview

```mermaid
flowchart TD
    A["Open-Meteo API<br/>(5 models × 85 stations)"] -->|async HTTP| B["API Cache<br/>(JSON, date-keyed)"]
    B --> C["Ensemble Statistics<br/>(weighted averaging)"]
    C --> D["Derived Indices<br/>(MCS, Heat, HI, Convergence)"]
    D --> E["RBF Interpolation<br/>(600×600 grid)"]
    E --> F["Terrain Correction<br/>(lapse rate projection)"]
    F --> G["Land Mask<br/>(Shapely polygon)"]
    G --> H["Map Rendering<br/>(Cartopy + matplotlib)"]

    I["ECMWF Open Data<br/>(GRIB2, 0.25°)"] -->|HTTP download| J["cfgrib parsing"]
    J --> K["GRIB Interpolation<br/>(xarray .interp)"]
    K --> L["Synoptic Products<br/>(MSLP, T2m, Convergence)"]
    L --> H

    H --> M["Output: forecasts_YYYY-MM-DD/<br/>~80+ PNG maps"]

    N["Open-Meteo Elevation API"] -->|batch fetch| O["Station + Anchor Elevations"]
    O --> F
```

### Pipeline Stages (in execution order)

| Stage | Description | Runtime |
|---|---|---|
| **Step 1** | Async fetch from Open-Meteo (5 models × 85 cities) + elevation API | ~15–30 s |
| **Step 2** | Compute ensemble stats, derived indices, interpolate & render 8 days × 9 map types | ~5–10 min |
| **Step 3** | Download ECMWF GRIB, extract T2m / MSLP / U10 / V10, render synoptic maps | ~1–2 min |

---

## 2. Station Network

The pipeline uses **85 observation points** across 5 South Indian states:

- **Tamil Nadu** — 42 stations (Chennai, Madurai, Coimbatore, Salem, Trichy, etc.)
- **Kerala** — 14 stations (Thiruvananthapuram, Kochi, Munnar, Wayanad, etc.)
- **Karnataka** — 9 stations (Bengaluru, Mysuru, Mangaluru, Agumbe, etc.)
- **Andhra Pradesh** — 5 stations (Tirupati, Nellore, Kadapa, Anantapur, Horsley Hills)
- **Puducherry** — 1 station

### Topographic Anchors

In addition to the 85 city stations, **24 topographic anchor points** are used for elevation interpolation:

- **6 mountain peaks** with known elevations (Kodaikanal 2695m, Ooty 2637m, Chikmagalur 1930m, etc.)
- **18 coastal zero-elevation anchors** along the west and east coasts

These anchors constrain the interpolated elevation grid so that terrain-corrected temperatures don't produce artifacts in data-sparse mountainous regions.

---

## 3. Data Acquisition

### 3.1 Open-Meteo Forecast API

For each of the 85 stations, the pipeline fetches **daily forecasts from 5 models** via the Open-Meteo API:

```
GET https://api.open-meteo.com/v1/forecast
    ?latitude={lat}&longitude={lon}
    &daily={10 variables}
    &models={model_key}
    &timezone=auto
    &forecast_days=10
```

**Variables fetched per model per station:**

| Variable | Unit | Description |
|---|---|---|
| `temperature_2m_max` | °C | Daily maximum 2m temperature |
| `temperature_2m_min` | °C | Daily minimum 2m temperature |
| `windspeed_10m_max` | km/h | Daily maximum sustained 10m wind speed |
| `windgusts_10m_max` | km/h | Daily maximum wind gusts at 10m |
| `precipitation_sum` | mm | Total daily precipitation |
| `cape_mean` | J/kg | Daily mean CAPE (Convective Available Potential Energy) |
| `dewpoint_2m_mean` | °C | Daily mean 2m dewpoint temperature |
| `vapour_pressure_deficit_max` | kPa | Daily max VPD |
| `winddirection_10m_dominant` | ° | Dominant wind direction (time-integrated) |
| `et0_fao_evapotranspiration` | mm | FAO Penman-Monteith reference ET |

**Total API calls:** 85 stations × 5 models = **425 requests** (run asynchronously via `aiohttp` with 20 connections per host).

### 3.2 Caching

All API data is cached to `api_cache_YYYY-MM-DD.json`. On re-runs within the same day, the cache is loaded if:
- The cache file exists
- It contains ≥ 90% of stations with data
- It includes the `et0_fao_evapotranspiration` variable (schema validation)

### 3.3 Elevation Data

Station elevations are fetched from the Open-Meteo Elevation API in batches of 10:

```
GET https://api.open-meteo.com/v1/elevation
    ?latitude={lats}&longitude={lons}
```

These are used for the sea-level temperature reduction (Section 6.2).

---

## 4. Ensemble Weighting

### 4.1 Models and Base Weights

| Model | Key | Weight | Type |
|---|---|---|---|
| ECMWF IFS 0.25° | `ecmwf_ifs025` | **35%** | Physics-based NWP |
| ECMWF AIFS 0.25° | `ecmwf_aifs025` | **25%** | AI/ML (graph neural network) |
| Google GraphCast | `google_graphcast_seamless` | **20%** | AI/ML (graph neural network) |
| DWD ICON Global | `icon_global` | **10%** | Physics-based NWP |
| NCEP GFS | `gfs_seamless` | **10%** | Physics-based NWP |

**Rationale:** ECMWF IFS is the consistently top-ranked global model in WMO scorecard verification. AIFS and GraphCast are the leading AI weather models. ICON and GFS provide diversity but with lower individual skill.

### 4.2 Variable-Specific Weights

Three variables have custom weight tables that override the base weights:

| Variable | Top Model | Rationale |
|---|---|---|
| **Precipitation** | ECMWF IFS (45%), ICON (25%) | Physics models handle precip better than AI models |
| **Temperature** | ECMWF AIFS (35%), GraphCast (30%) | AI models excel at smooth fields like temperature |
| **Wind Gusts** | ECMWF IFS (40%), ICON (30%) | Convective gusts need explicit parameterization |

All other variables (CAPE, dewpoint, VPD, ET0, wind speed) use the base weight table.

### 4.3 Averaging Procedure

For each variable at each station on each forecast day:

1. Collect values from all models that returned valid data
2. Look up variable-specific weights (or fall back to base weights)
3. **Re-normalize** weights to sum to 1.0 (handles missing models gracefully)
4. Compute weighted average: `ensemble = Σ(wᵢ × valᵢ) / Σ(wᵢ)`

Additionally computed:
- **Spread** = max(values) − min(values) — inter-model disagreement
- **N models** = count of models with valid data

### 4.4 Wind Direction Averaging

Wind direction cannot be arithmetically averaged (359° and 1° would average to 180°, not 0°). Instead:

1. **Decompose** each model's wind into (u, v) components using speed and direction
2. **Scale** speed by ×0.50 to approximate daily mean wind (since `windspeed_10m_max` is a daily maximum — see [Section 10](#10-known-approximations))
3. **Average** u and v components independently across models
4. **Reconstruct** mean direction via `atan2(-u_mean, -v_mean)`

The resulting `u_wind` and `v_wind` fields are stored for convergence calculation.

---

## 5. Derived Variables & Indices

### 5.1 MCS (Mesoscale Convective System) Score — 0 to 10

A composite index estimating severe thunderstorm / squall line potential:

```
Score = comp_precip + comp_gusts + comp_cape    (clamped to [0, 10])

comp_precip = min(4.0, P90 / 12.5)             if P90 > 1 mm, else 0
comp_gusts  = min(3.0, max(0, (gusts − 30) / 13.3))
comp_cape   = min(3.0, max(0, (CAPE − 200) / 766))
```

| Score | Tier | Meaning |
|---|---|---|
| 0 | Nil | No significant convective risk |
| 1–2 | Isolated Convection | Scattered showers possible |
| 3–4 | Organized Storms | Moderate thunderstorm clusters |
| 5–6 | Squall Line | Strong organized convection |
| 7–8 | Severe MCS | Damaging winds, heavy rain |
| 9–10 | Extreme | Catastrophic convective event |

**NaN Safety:** If any input (P90, gusts, CAPE) is NaN, it is treated as 0 (zero risk contribution).

### 5.2 Heat Score — 0 to 10

A composite index for heatwave severity:

```
Score = comp_temp + comp_vpd + comp_dewpoint    (clamped to [0, 10])

comp_temp     = min(4.0, max(0, (Tmax − 35) / 1.5))
comp_vpd      = min(3.0, max(0, (VPD − 1.5) / 0.83))
comp_dewpoint = min(3.0, max(0, (Dp − 24) / 2.0))
```

| Score | Tier | Meaning |
|---|---|---|
| 0–1 | Normal | No heat stress |
| 2–3 | Watch | Elevated temperatures, stay hydrated |
| 4–5 | Heatwave | Official heatwave threshold |
| 6–7 | Severe Heatwave | Dangerous heat stress |
| 8–10 | Extreme | Life-threatening conditions |

> [!NOTE]
> VPD and dewpoint both measure atmospheric moisture from different perspectives. In humid
> tropical heat (South India's primary hazard), both components score high. This is an
> intentional design choice — humid heat is disproportionately dangerous compared to dry heat.

### 5.3 Heat Index (Apparent Temperature)

The NWS Rothfusz regression — the standard used by the US National Weather Service:

1. **Estimate RH** from Tmax and dewpoint using the Magnus formula:
   ```
   RH = 100 × exp(17.625 × Dp / (243.04 + Dp)) / exp(17.625 × T / (243.04 + T))
   ```

2. **Simple formula** (first pass):
   ```
   HI = 0.5 × (T_F + 61.0 + (T_F − 68.0) × 1.2 + RH × 0.094)
   ```

3. **Full Rothfusz regression** (if simple HI ≥ 80°F):
   ```
   HI = −42.379 + 2.049T + 10.143RH − 0.225T·RH − 0.007T² − 0.055RH²
        + 0.001T²·RH + 0.001T·RH² − 0.000002T²·RH²
   ```

4. **Adjustments:**
   - Low RH correction (RH < 13%, 80 ≤ T_F ≤ 112)
   - High RH correction (RH > 85%, 80 ≤ T_F ≤ 87)

5. **Floor:** `max(Tmax, HI_celsius)` — heat index cannot be below air temperature.

### 5.4 Diurnal Temperature Range (DTR)

Simply: `DTR = Tmax − Tmin`

Used both as a map product and as an input to the dynamic lapse rate.

### 5.5 Precipitation Percentiles

From the raw model outputs (before weighting):
- **P90** = 90th percentile across models — conservative upper-bound estimate
- **P_max** = maximum across models
- **P_prob** = fraction of models predicting > 1mm × 100%

---

## 6. Spatial Interpolation & Terrain Correction

### 6.1 RBF Interpolation

All fields are interpolated from the 85 station points to a **600 × 600 regular grid** using **Radial Basis Function (RBF) interpolation** (`scipy.interpolate.RBFInterpolator`).

- **Grid extent:** Covers the land boundary of the 5 target states ± 0.3° padding
- **Northern boundary:** Capped at 14.90°N
- **Grid resolution:** ~0.012° per pixel (≈1.3 km)
- **Kernel:** `linear` for most variables, `thin_plate_spline` for wind u/v (convergence)

### Land Masking

A polygon mask is constructed from Natural Earth 10m admin boundaries for the 5 target states. After interpolation, all grid cells outside the land polygon are set to NaN. This prevents spurious offshore values.

### 6.2 Sea-Level Temperature Reduction

Directly interpolating temperature across elevation gradients produces severe artifacts (a station at 2000m reporting 18°C would pull down temperatures in nearby lowland areas). The pipeline uses a **reduce-to-sea-level → interpolate → project-back** workflow:

#### Daytime (MaxTemp):

```
1. T_sea_level = T_station + (elevation_km × dynamic_lapse_rate)
2. Interpolate T_sea_level across 600×600 grid
3. T_grid = T_sea_level_grid − (grid_elevation_km × interpolated_lapse_rate)
```

The **dynamic lapse rate** is calculated per station based on DTR:
```
lapse = clip(5.0 + (DTR − 5) × 4.8/5.0, min=5.0, max=9.8) °C/km
```

**Physical rationale:**
- High DTR → dry, well-mixed boundary layer → lapse rate approaches dry adiabatic (9.8 °C/km)
- Low DTR → moist, stable conditions → lapse rate closer to saturated adiabatic (5–6 °C/km)

#### Nighttime (MinTemp):

Uses a **fixed lapse rate of 3.0 °C/km** — reflecting nocturnal radiative inversions and stable nighttime boundary layers common in tropical environments.

### 6.3 Elevation Grid

The elevation grid `gelev` is itself RBF-interpolated from:
- 85 station elevations (from the Open-Meteo elevation API)
- 24 topographic anchor points (6 mountain peaks + 18 coastal zeros)

Clamped to ≥ 0 to prevent negative elevations.

---

## 7. ECMWF GRIB Automation

After the ensemble maps, the pipeline downloads **one ECMWF IFS operational GRIB2 file** from the ECMWF Open Data service for high-resolution synoptic analysis.

### 7.1 Cycle Selection Logic

| UTC Hour | Model Cycle | Step | Valid Time |
|---|---|---|---|
| 00–09 UTC | Previous day 12z | +24h | 12z today |
| 10–20 UTC | Today 00z | +12h | 12z today |
| 21–23 UTC | Today 12z | +24h | 12z tomorrow |

This ensures the pipeline always targets a **12:00 UTC valid time** snapshot (≈17:30 IST), which represents afternoon peak conditions.

### 7.2 Variables Extracted

| GRIB Variable | Product | Processing |
|---|---|---|
| `2t` (2m temperature) | Raw temperature snapshot map | Convert K → °C if needed |
| `msl` (mean sea level pressure) | Synoptic MSLP + LPA map | Convert Pa → hPa |
| `10u` (10m u-wind) | Convergence snapshot | Direct use (m/s) |
| `10v` (10m v-wind) | Convergence snapshot | Direct use (m/s) |

### 7.3 LPA (Low Pressure Area) Detection

Applied to the MSLP field:

1. **Minimum filter** with 150-pixel (~180 km) neighborhood — isolates synoptic-scale minima
2. **Threshold** at < 1010 hPa — only genuine low-pressure systems
3. **Deduplication** — merge detections within 160 km of each other
4. Mark with red "**L**" symbol and pressure value on the map

### 7.4 ECMWF Contoured Convergence

The GRIB u10/v10 fields are used to compute a **true grid-scale convergence** (as opposed to the ensemble convergence which is interpolated from station points):

- Physical dx/dy scaling identical to the ensemble version
- Gaussian smoothing with σ=2.0 (lighter than ensemble's σ=8.0, since the 0.25° grid is already coherent)
- Rendered as **contour lines** (not filled) at levels [1.5, 3.0, 5.0, 8.0, 12.0] ×10⁻⁵ s⁻¹

---

## 8. Map Rendering & Output Products

### 8.1 Map Layout

Every map uses an identical layout framework:

```
┌──────────────────────────────────────────┐
│  HEADER (1.35")                          │
│  Title · Date · Model Label String       │
├─────────────────────────────────┬────────┤
│                                 │        │
│                                 │ Color  │
│         MAP PANEL               │  Bar   │
│      (Cartopy PlateCarree)      │(1.00") │
│      Google Satellite Base      │        │
│                                 │        │
├─────────────────────────────────┴────────┤
│  FOOTER (0.85")                          │
│  Top 5 places · Disclaimer · Timestamp   │
└──────────────────────────────────────────┘
```

- **Base layer:** Google satellite imagery (zoom level 7)
- **Overlays:** State boundaries (dotted), coastline, country borders
- **Color bar:** Vertical, to the right of the map panel
- **DPI:** 200

### 8.2 Output Map Catalog

For **each of the 8 forecast days** (today + 7 days), the following maps are generated:

| # | Filename Pattern | Variable | Colormap | Scale |
|---|---|---|---|---|
| 01 | `01_maxtemp_T{n}_{date}.png` | Peak Temperature | `magma_r` | 24–44 °C |
| 02 | `02_precip_T{n}_{date}.png` | Precipitation P90 | `viridis` | LogNorm, ≥1 mm |
| 03 | `03_mcs_T{n}_{date}.png` | MCS / Thundersquall Index | Custom 6-tier | 0–10 (categorical) |
| 04 | `04_gusts_T{n}_{date}.png` | Wind Gusts | `plasma_r` | 0–80 km/h |
| 05a | `05a_heat_index_T{n}_{date}.png` | Apparent Temperature | `OrRd` | 30–50 °C |
| 05b | `05b_heatwave_T{n}_{date}.png` | Heatwave Alert Index | Custom 5-tier | 0–10 (categorical) |
| 06 | `06_dtr_T{n}_{date}.png` | Diurnal Temperature Range | `turbo` | 2–18 °C |
| 07 | `07_mintemp_T{n}_{date}.png` | Nighttime Min Temperature | `inferno` | 18–32 °C |
| 08 | `08_et0_T{n}_{date}.png` | Evapotranspiration ET₀ | `YlOrBr` | 2–8 mm/day |
| 09 | `09_convergence_T{n}_{date}.png` | Wind Convergence | `PiYG` | ±5.0 ×10⁻⁵ s⁻¹ |

**Plus one-time synoptic products:**

| # | Filename | Source | Description |
|---|---|---|---|
| 00 | `00_google_terrain_basemap.png` | — | Clean reference map |
| 00 | `00_ecmwf_raw_t2m.png` | GRIB | ECMWF temperature snapshot |
| 00 | `00_ecmwf_mslp_lpa.png` | GRIB | MSLP isobars + LPA markers |
| 00 | `00_ecmwf_convergence_snapshot.png` | GRIB | True-grid convergence contours |

**Total output:** ~80+ maps per run (8 days × 9 types + 4 synoptic).

### 8.3 City Labels

- **Collision avoidance:** Labels are placed only if no other label is within 0.55° (~60 km)
- **Priority cities:** Gudiyatham, Vaithiyampettai, and Madurai are always shown regardless of crowding
- **Edge clipping:** Labels within 0.25° of the map boundary are suppressed to prevent overflow
- **Styling:** Bold text with white stroke outline for legibility over satellite imagery

### 8.4 Top 5 Footer

Each map includes a "Top 5" line in the footer showing the cities with the highest values for the displayed variable.

---

## 9. Convergence Calculation

Wind convergence is a key diagnostic for storm initiation. The pipeline computes it in two contexts:

### 9.1 Mathematical Definition

Horizontal wind convergence is defined as the negative divergence of the wind field:

```
Convergence = −(∂u/∂x + ∂v/∂y)
```

- **Positive** values → air is converging (piling up) → forced lift → potential storm initiation
- **Negative** values → air is diverging → suppressed convection

### 9.2 Physical Scaling

Raw `np.gradient()` returns derivatives in **index space** (per grid cell). These must be converted to physical units:

```
dx = Δlon_degrees × 111,139 m × cos(latitude)    [meters per grid cell in x]
dy = Δlat_degrees × 111,139 m                     [meters per grid cell in y]

∂u/∂x = (∂u/∂i) / dx     [s⁻¹]
∂v/∂y = (∂v/∂j) / dy     [s⁻¹]
```

The `cos(latitude)` factor accounts for meridian convergence — at 10°N, 1° of longitude is ~109.5 km, while at 14°N it's ~107.9 km.

### 9.3 Units

Results are scaled by ×100,000 and presented in **10⁻⁵ s⁻¹**, which is the standard meteorological unit for synoptic/mesoscale convergence. Typical values:

| Value (×10⁻⁵ s⁻¹) | Interpretation |
|---|---|
| < 0 | Divergence (fair weather) |
| 0–1.5 | Weak, negligible forcing |
| 1.5–3.0 | Moderate convergence |
| 3.0–5.0 | Significant — potential storm trigger |
| 5.0–8.0 | Strong — organized convection likely |
| > 8.0 | Intense — MCS / squall initiation zone |

### 9.4 Ensemble vs. ECMWF Convergence

| Aspect | Ensemble (daily) | ECMWF (GRIB snapshot) |
|---|---|---|
| Source | 85 station u/v → RBF interpolated | Native 0.25° grid u/v |
| Wind units | km/h → m/s (÷ 3.6) | Already m/s |
| Smoothing | Gaussian σ=8 pixels | Gaussian σ=2.0 pixels |
| Rendering | Filled colormap (PiYG) | Contour lines only |
| Scale factor | ×0.50 (mean wind approx.) | None needed (instantaneous) |

---

## 10. Known Approximations

### 10.1 Wind Component Synthesis (v19.3 fix applied)

The daily API provides `windspeed_10m_max` (a daily maximum) and `winddirection_10m_dominant` (a time-integrated prevailing direction). These quantities do not come from the same instant — the max gust might occur during a brief squall while the dominant direction reflects the prevailing trade winds.

**Mitigation (v19.3):** The speed is scaled by **×0.50** before u/v decomposition, approximating a daily mean sustained wind. This is based on the empirical ratio of daily mean to daily maximum wind speed in tropical environments (Justus et al. 1978; WMO-No. 8). The resulting convergence magnitudes are physically plausible, though the spatial patterns may miss sub-daily wind shifts.

### 10.2 Composite Score Interpolation

MCS Score and Heat Score are **nonlinear composites** of multiple variables. The pipeline interpolates the computed score directly (rather than interpolating component variables first and then computing the score on the grid). This can produce intermediate values between stations that are physically unrealistic — for instance, averaging a "high precip, low CAPE" score with a "low precip, high CAPE" score may overstate risk at the midpoint.

### 10.3 Day 0 (Current Day) Forecast

The current-day forecast includes data that may partially be observed (analysis) rather than pure forecast, depending on the model initialization time. Open-Meteo blends these seamlessly. Maps for day 0 should be interpreted as "best estimate for today" rather than a pure forecast.

### 10.4 Precipitation Non-Negativity

RBF interpolation does not enforce non-negativity. Precipitation values below 1.0 mm are masked to NaN for display, which hides any negative overshoot. The underlying grid may contain physically impossible negative values between widely-spaced stations with large precipitation gradients.
