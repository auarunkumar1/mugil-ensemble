# MUGIL Ensemble Weather Forecasting Pipeline — v19.4 (Corrected)
# ========================================================
# Generates interpolated meteorological maps for South India from
# multiple NWP and AI models via Open-Meteo & ECMWF Open Data.
#
# v19.4 Upgrades:
#   1. GOOGLE WEATHERNEXT 2 INTEGRATION: Replaced deprecated GraphCast (google_graphcast_seamless)
#      with Google DeepMind WeatherNext 2 (google_weathernext2_ensemble) via Open-Meteo ensemble endpoint.
#   2. ECMWF AIFS ENDPOINT ROUTING: Routed ecmwf_aifs025 to Open-Meteo ensemble endpoint
#      (ensemble-api.open-meteo.com) resolving empty/null forecasts from deprecated forecast endpoint path.
#   3. OCEANIC SST DIAGNOSTICS: Integrated high-resolution NOAA OISST v2.1 maps for the
#      Indian Ocean Dipole (IOD: Arabian Sea & Bay of Bengal) and Pacific Basin (ENSO Niño 1+2, 3, 3.4, 4).
#
# v19.3 Upgrades:
#   1. SEA-LEVEL TEMPERATURE REDUCTION: Station temperatures reduced to sea level
#      before RBF interpolation, then projected back to terrain (using 3.0°C/km night lapse).
#   2. PHYSICAL SCALE CONVERGENCE: Convergence calculations scaled using actual 
#      physical distance (meters) based on grid latitude, and scaled to 10^-5 s^-1.
#   3. GFS WEIGHT BOOST REMOVED: Reverted dynamic weighting to maintain ensemble stability.
#   4. HEAT INDEX REVISION: Standard NWS Rothfusz regression with high/low RH adjustments.
#   5. SYNOPTIC LPA DETECTION: Increased minimum filter to 150 pixels (~180km) to capture true LPAs.
#   6. ECMWF GRIB LEAD TIME CONSISTENCY: Snapshot valid hours aligned to exactly 12:00 UTC.

import logging
import warnings
import numpy as np
import pandas as pd
import requests
import time
import os
import json
import asyncio
import aiohttp
import math
from datetime import datetime, timedelta, timezone
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
from matplotlib.ticker import ScalarFormatter

import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import cartopy.io.img_tiles as cimgt
from shapely.ops import unary_union
import scipy.ndimage as ndimage
from scipy.interpolate import RBFInterpolator

# [Antigravity Windows Patch] Monkeypatch findlibs to support Windows Conda layout for ecCodes
try:
    import findlibs
    import sys
    import os
    orig_find = findlibs.find
    def patched_find(lib_name, pkg_name=None):
        if lib_name == "eccodes" and sys.platform == "win32":
            # Search under sys.prefix/Library/bin/eccodes.dll
            p = os.path.join(sys.prefix, "Library", "bin", "eccodes.dll")
            if os.path.exists(p):
                return p
            cp = os.environ.get("CONDA_PREFIX")
            if cp:
                p = os.path.join(cp, "Library", "bin", "eccodes.dll")
                if os.path.exists(p):
                    return p
        return orig_find(lib_name, pkg_name)
    findlibs.find = patched_find
except Exception as e:
    pass

try:
    import cfgrib  # noqa: F401
    _CFGRIB_OK = True
except Exception:
    _CFGRIB_OK = False

# ── 1. INITIALIZATION ─────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s",
                    handlers=[logging.StreamHandler()])
log = logging.getLogger(__name__)

# Base weights used as a global fallback
MODEL_WEIGHTS = {
    "ecmwf_ifs025":            0.35,
    "ecmwf_aifs025":           0.25,
    "google_weathernext2_ensemble": 0.20,
    "icon_global":             0.10,
    "gfs_seamless":            0.10
}

VAR_WEIGHTS = {
    "precipitation_sum":           {"ecmwf_ifs025": 0.45, "icon_global": 0.25, "ecmwf_aifs025": 0.15, "google_weathernext2_ensemble": 0.10, "gfs_seamless": 0.05},
    "temperature_2m_max":          {"ecmwf_aifs025": 0.35, "google_weathernext2_ensemble": 0.30, "ecmwf_ifs025": 0.20, "icon_global": 0.10, "gfs_seamless": 0.05},
    "windgusts_10m_max":           {"ecmwf_ifs025": 0.40, "icon_global": 0.30, "ecmwf_aifs025": 0.15, "google_weathernext2_ensemble": 0.10, "gfs_seamless": 0.05},
}

import functools
print = functools.partial(print, flush=True)
warnings.filterwarnings("ignore")

RUN_DATE_STR = datetime.now().strftime("%Y-%m-%d")
RUN_TIMESTAMP = datetime.now().strftime("%d %b %Y  %H:%M IST")
OUT_DIR = f"forecasts_{RUN_DATE_STR}"
CACHE_FILE = f"api_cache_{RUN_DATE_STR}.json"
os.makedirs(OUT_DIR, exist_ok=True)

forecast_days = [datetime.now() + timedelta(days=d) for d in range(0, 8)]

ENSEMBLE_MODELS = {
    "ecmwf_ifs025":            "ECMWF IFS (35%)",
    "ecmwf_aifs025":           "ECMWF AIFS (25%)",
    "google_weathernext2_ensemble": "WeatherNext 2 (20%)",
    "icon_global":             "ICON Global (10%)",
    "gfs_seamless":            "GFS (10%)",
}
MODEL_LABEL_STR = " · ".join(ENSEMBLE_MODELS.values())

DAILY_VARS = [
    "temperature_2m_max", "temperature_2m_min", "windspeed_10m_max",
    "windgusts_10m_max", "precipitation_sum", "cape_mean",
    "dewpoint_2m_mean", "vapour_pressure_deficit_max",
    "winddirection_10m_dominant", "et0_fao_evapotranspiration"
]

# ── 2. HTTP SESSION ───────────────────────────────────────────────────────────
def make_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "MUGIL-Ensemble-Pipeline/19.4 (research; non-commercial)"})
    retry_codes = list((429, 500, 502, 503, 504))
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=retry_codes)
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

SESSION = make_session()

# ── 3. LOCATIONS & ANCHORS ────────────────────────────────────────────────────
locations = {
    "Kodaikanal":       (10.2381, 77.4892), "Ooty":          (11.4100, 76.6950),
    "Valparai":         (10.3276, 76.9554), "Yercaud":       (11.7750, 78.2017),
    "Meghamalai":       ( 9.5852, 77.4124), "Kolli Hills":   (11.2485, 78.3385),
    "Javadhu Hills":    (12.5933, 78.8953), "Sirumalai":     (10.2115, 77.9945),
    "Chennai":          (13.0827, 80.2707), "Pondicherry":   (11.9416, 79.8083),
    "Cuddalore":        (11.7480, 79.7714), "Nagapattinam":  (10.7672, 79.8444),
    "Thoothukudi":      ( 8.7139, 78.1348), "Kanyakumari":   ( 8.0883, 77.5385),
    "Vaithiyampettai":  (11.2076, 79.6229), "Velankanni":    (10.6823, 79.8488),
    "Ramanathapuram":   ( 9.3678, 78.8305), "Madurai":       ( 9.9252, 78.1198),
    "Coimbatore":       (11.0168, 76.9558), "Salem":         (11.6643, 78.1460),
    "Tiruchirappalli":  (10.7905, 78.7047), "Tirunelveli":   ( 8.7139, 77.7567),
    "Pollachi":         (10.6580, 77.0102), "Karur":         (10.9601, 78.0766),
    "Erode":            (11.3410, 77.7172), "Vellore":       (12.9165, 79.1325),
    "Gudiyatham":       (12.9474, 78.8712), "Dharmapuri":    (12.1273, 78.1582),
    "Thanjavur":        (10.7870, 79.1378), "Dindigul":      (10.3673, 77.9803),
    "Theni":            (10.0104, 77.4768), "Hosur":         (12.7409, 77.8253),
    "Villupuram":       (11.9459, 79.4974), "Kanchipuram":   (12.8461, 79.7119),
    "Virudhachalam":    (11.5067, 79.0253), "Kamudhi":       ( 9.4042, 78.3635),
    "Thiruvannamalai":  (12.2404, 79.0772), "Mimisal":       ( 9.9239, 79.1468),
    "Kovilpatti":       ( 9.1626, 77.8273), "Kallakurichi":  (11.7438, 78.9413),
    "Thiruvananthapuram":(8.4855, 76.9492), "Kochi":         ( 9.9312, 76.2673),
    "Munnar":           (10.0889, 77.0595), "Idukki":        ( 9.8500, 76.9700),
    "Wayanad":          (11.6854, 76.1320), "Kozhikode":     (11.2588, 75.7804),
    "Bengaluru":        (12.9716, 77.5946), "Mysuru":        (12.2958, 76.6394),
    "Mangaluru":        (12.9141, 74.8560), "Agumbe":        (13.5076, 75.0955),
    "Madikeri":         (12.4244, 75.7382), "Chikmagalur":   (13.3161, 75.7720),
    "Tirupati":         (13.6288, 79.4192), "Nellore":       (14.4426, 79.9865),
    "Kadapa":           (14.4673, 78.8242), "Anantapur":     (14.6819, 77.6006),
    "Horsley Hills":    (13.6500, 78.4000), "Thrissur":      (10.5276, 76.2144),
    "Palakkad":         (10.7867, 76.6548), "Alappuzha":     ( 9.4981, 76.3388),
    "Kollam":           ( 8.8932, 76.6141), "Kasaragod":     (12.4996, 74.9869),
    "Kannur":           (11.8745, 75.3704), "Malappuram":    (11.0510, 76.0711),
    "Tenkasi":          ( 8.9597, 77.3152), "Sivakasi":      ( 9.4530, 77.7974),
    "Kumbakonam":       (10.9602, 79.3845), "Chidambaram":   (11.3990, 79.6935),
    "Pudukkottai":      (10.3797, 78.8201), "Namakkal":      (11.2189, 78.1674),
    "Tiruvallur":       (13.1431, 79.9079), "Ranipet":       (12.9226, 79.3326),
    "Perambalur":       (11.2342, 78.8808), "Ariyalur":      (11.1434, 79.0761),
    "Shivamogga":       (13.9299, 75.5681), "Hassan":        (13.0068, 76.1004),
    "Udupi":            (13.3409, 74.7421), "Tumkur":        (13.3392, 77.1010),
}

topo_anchors = [
    (10.1667, 77.0667, 2695), (11.4000, 76.7333, 2637),
    (13.3833, 75.7167, 1930), ( 8.6167, 77.2500, 1868),
    (13.6833, 79.3500,  970), (12.5700, 78.6300, 1110),
    ( 8.2, 77.0, 0), ( 8.5, 76.9, 0), ( 9.0, 76.5, 0), ( 9.5, 76.3, 0),
    (10.0, 76.2, 0), (10.5, 75.9, 0), (11.0, 75.8, 0), (11.5, 75.4, 0),
    (12.0, 75.2, 0), (12.5, 74.9, 0), (13.0, 74.7, 0), (13.5, 74.6, 0),
    (14.0, 74.4, 0), (13.0, 80.3, 0), (12.0, 79.9, 0), (11.0, 79.8, 0),
    (10.0, 79.3, 0), ( 9.0, 78.5, 0),
]

def get_elevations(coords_list, batch_size=10):
    result = [0.0] * len(coords_list)
    for start in range(0, len(coords_list), batch_size):
        chunk = coords_list[start:start + batch_size]
        try:
            lats = ",".join(str(lat_val) for lat_val, lon_val in chunk)
            lons = ",".join(str(lon_val) for lat_val, lon_val in chunk)
            resp = SESSION.get(
                f"https://api.open-meteo.com/v1/elevation?latitude={lats}&longitude={lons}",
                timeout=15
            ).json()
            for k, v in enumerate(resp.get("elevation", [])):
                result[start + k] = float(v) if v is not None else 0.0
        except Exception as e:
            log.warning(f"  Elevation fetch failed for chunk starting at {start}: {e}")
        time.sleep(0.05)
    return result

# ── 4. ASYNC FORECAST FETCHING & CACHING ──────────────────────────────────────
_API_SEM = None  # initialized per-run in fetch_all_cities_async; caps concurrent Open-Meteo requests

async def fetch_one_model_async(session, lat, lon, model_key, city_name, retries=3):
    base_url = "https://ensemble-api.open-meteo.com/v1/ensemble" if any(k in model_key for k in ("ensemble", "weathernext", "aifs")) else "https://api.open-meteo.com/v1/forecast"
    url = (f"{base_url}"
           f"?latitude={lat}&longitude={lon}"
           f"&daily={','.join(DAILY_VARS)}"
           f"&models={model_key}&timezone=auto&forecast_days=10")

    for attempt in range(retries):
        try:
            async with _API_SEM:
                async with session.get(url, timeout=20) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(0.5 * (2 ** attempt))
                        continue

                    r = await resp.json()
                    if "error" in r or "daily" not in r:
                        return city_name, model_key, {}

                    daily = r["daily"]
                    times = daily.get("time", [])
                    out = {}
                    for dt in forecast_days:
                        dt_str = dt.strftime("%Y-%m-%d")
                        if dt_str in times:
                            idx = times.index(dt_str)
                            row = {}
                            for var in DAILY_VARS:
                                col_data = daily.get(var)
                                col_val = col_data[idx] if col_data is not None else None
                                row[var] = float(col_val) if col_val is not None else np.nan
                            out[dt_str] = row
                    return city_name, model_key, out

        except Exception as e:
            if attempt == retries - 1:
                log.warning(f"  Fetch failed for {city_name} ({model_key}) after {retries} attempts: {e}")
            await asyncio.sleep(0.5)

    return city_name, model_key, {}

async def fetch_all_cities_async(locs):
    global _API_SEM
    _API_SEM = asyncio.Semaphore(10)  # max 10 concurrent requests to Open-Meteo
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cached_data = json.load(f)
            sample_city = next(iter(cached_data.values()), {})
            sample_day = next(iter(sample_city.values()), {})
            sample_model = next(iter(sample_day.values()), {})
            if "et0_fao_evapotranspiration" in sample_model and sum(1 for c_data in cached_data.values() if any(day_data for day_data in c_data.values())) >= len(locs) * 0.9:
                log.info("  📦 Loading API data from local cache...")
                return cached_data

    log.info("  🌐 Fetching global ensemble asynchronously (this is fast)...")
    results = {city: {dt.strftime("%Y-%m-%d"): {} for dt in forecast_days} for city in locs}

    connector = aiohttp.TCPConnector(limit_per_host=10, limit=50)  # fair-use: ≤10 concurrent per host
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for city_name, (lat, lon) in locs.items():
            for mk in ENSEMBLE_MODELS:
                tasks.append(fetch_one_model_async(session, lat, lon, mk, city_name))

        responses = await asyncio.gather(*tasks)

        for city_name, mk, model_out in responses:
            for dt_str, vals in model_out.items():
                if dt_str in results[city_name]:
                    results[city_name][dt_str][mk] = vals

    successful_cities = sum(1 for c_data in results.values() if any(day_data for day_data in c_data.values()))
    if successful_cities >= len(locs) * 0.9:
        with open(CACHE_FILE, "w") as f:
            json.dump(results, f)

    return results

# ── 5. STATS & SCORING ────────────────────────────────────────────────────────
def compute_ensemble_stats(model_data, day_idx):
    stats = {}

    # windspeed_10m_max is a daily MAXIMUM, not a mean.  Pairing it with
    # winddirection_10m_dominant (a time-integrated prevailing direction)
    # produces an artificially strong synthetic vector.  We scale the speed
    # down by 0.50 to approximate the daily mean sustained wind, which is
    # the quantity physically appropriate for convergence (∇·V).
    # Ratio of daily mean to daily max sustained wind ≈ 0.4–0.6 in tropical
    # / subtropical environments (Justus et al. 1978; WMO-No. 8).
    # NOTE: The direction from atan2 is scale-invariant, so this only
    # affects the magnitude of u_wind / v_wind used for convergence maps.
    MEAN_WIND_SCALE = 0.50

    u_vals, v_vals = [], []
    for mk in model_data:
        spd = model_data[mk].get("windspeed_10m_max")
        wdir = model_data[mk].get("winddirection_10m_dominant")
        if pd.notna(spd) and pd.notna(wdir):
            representative_spd = spd * MEAN_WIND_SCALE
            u = -representative_spd * math.sin(math.radians(wdir))
            v = -representative_spd * math.cos(math.radians(wdir))
            u_vals.append(u)
            v_vals.append(v)

    if u_vals:
        u_mean = np.mean(u_vals)
        v_mean = np.mean(v_vals)
        mean_dir = (math.degrees(math.atan2(-u_mean, -v_mean)) + 360) % 360
        stats["winddirection_10m_dominant"] = mean_dir
        stats["u_wind"] = u_mean
        stats["v_wind"] = v_mean
    else:
        stats["winddirection_10m_dominant"] = np.nan
        stats["u_wind"] = np.nan
        stats["v_wind"] = np.nan

    for var in DAILY_VARS:
        if var == "winddirection_10m_dominant":
            continue

        vals, weights = [], []
        for mk in model_data:
            if var not in model_data[mk] or np.isnan(model_data[mk][var]):
                continue

            val = model_data[mk][var]
            final_w = VAR_WEIGHTS.get(var, MODEL_WEIGHTS).get(mk, MODEL_WEIGHTS.get(mk, 0.20))

            vals.append(val)
            weights.append(final_w)

        if vals:
            w_sum = sum(weights)
            if w_sum > 0:
                nw = np.array(weights) / w_sum
                stats[var] = float(np.average(vals, weights=nw))
            else:
                stats[var] = float(np.nanmean(vals))

            stats[var + "_spread"]  = float(np.max(vals) - np.min(vals))
            stats[var + "_nmodels"] = len(vals)
        else:
            stats[var]              = np.nan
            stats[var + "_spread"]  = np.nan
            stats[var + "_nmodels"] = 0

    p_vals = [model_data[mk]["precipitation_sum"] for mk in model_data
              if "precipitation_sum" in model_data[mk]
              and not np.isnan(model_data[mk]["precipitation_sum"])]

    if p_vals:
        stats["precip_p90"]  = float(np.percentile(p_vals, 90))
        stats["precip_max"]  = float(np.max(p_vals))
        stats["precip_prob"] = 100.0 * sum(1 for v in p_vals if v > 1.0) / len(p_vals)
    else:
        stats["precip_p90"]  = np.nan
        stats["precip_max"]  = np.nan
        stats["precip_prob"] = np.nan

    return stats

def compute_mcs_score(stats):
    p     = stats.get("precip_p90",  0)
    g     = stats.get("windgusts_10m_max",  0)
    cape  = stats.get("cape_mean",          0)

    # Guard against NaN propagation — treat missing data as zero risk
    p    = p    if pd.notna(p)    else 0
    g    = g    if pd.notna(g)    else 0
    cape = cape if pd.notna(cape) else 0

    comp_a = min(4.0, (p / 12.5)) if p > 1 else 0
    comp_b = min(3.0, max(0, (g - 30) / 13.3))
    comp_c = min(3.0, max(0, (cape - 200) / 766.0))

    return float(np.clip(comp_a + comp_b + comp_c, 0, 10))

def compute_heat_score(stats):
    tmax = stats.get("temperature_2m_max", 0)
    vpd  = stats.get("vapour_pressure_deficit_max", 0)
    dp   = stats.get("dewpoint_2m_mean", 0)

    # Guard against NaN propagation — treat missing data as zero risk
    tmax = tmax if pd.notna(tmax) else 0
    vpd  = vpd  if pd.notna(vpd)  else 0
    dp   = dp   if pd.notna(dp)   else 0

    comp_t = min(4.0, max(0, (tmax - 35) / 1.5))
    comp_v = min(3.0, max(0, (vpd - 1.5) / 0.83))
    comp_d = min(3.0, max(0, (dp - 24) / 2.0))

    return float(np.clip(comp_t + comp_v + comp_d, 0, 10))

def compute_heat_index(tmax, dp):
    if np.isnan(tmax) or np.isnan(dp):
        return np.nan
    rh = 100 * math.exp((17.625 * dp) / (243.04 + dp)) / math.exp((17.625 * tmax) / (243.04 + tmax))
    rh = np.clip(rh, 0, 100)
    tf = (tmax * 9/5) + 32
    
    # NWS simple formula
    hi = 0.5 * (tf + 61.0 + ((tf - 68.0) * 1.2) + (rh * 0.094))
    
    # If simple index is 80°F or higher, use full regression
    if hi >= 80:
        hi = (-42.379 + 2.04901523*tf + 10.14333127*rh - 0.22475541*tf*rh
              - 0.00683783*tf*tf - 0.05481717*rh*rh + 0.00122874*tf*tf*rh
              + 0.00085282*tf*rh*rh - 0.00000199*tf*tf*rh*rh)
        
        # Adjustments
        if rh < 13 and 80 <= tf <= 112:
            adjustment = ((13 - rh)/4) * math.sqrt((17 - abs(tf - 95.0))/17)
            hi -= adjustment
        elif rh > 85 and 80 <= tf <= 87:
            adjustment = ((rh - 85)/10) * ((87 - tf)/5)
            hi += adjustment
            
    hi_c = (hi - 32) * 5/9
    return max(tmax, hi_c)

# ── 6. CARTOPY MAP SETUP & RBF INTERPOLATION ──────────────────────────────────
class GoogleTerrain(cimgt.GoogleTiles):
    def _image_url(self, tile):
        x, y, z = tile
        return f'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'

shpfilename = shpreader.natural_earth(resolution='10m', category='cultural',
                                      name='admin_1_states_provinces')
reader = shpreader.Reader(shpfilename)
target_states = ["Tamil Nadu", "Kerala", "Karnataka", "Andhra Pradesh", "Puducherry"]

region_geoms = [
    rec.geometry for rec in reader.records()
    if rec.attributes['admin'] == 'India'
    and any(ts in rec.attributes.get('name', '') for ts in target_states)
]
land_geom = unary_union(region_geoms)

coords   = list(locations.values())
lats_arr = np.array([lat_val for lat_val, lon_val in coords])
lons_arr = np.array([lon_val for lat_val, lon_val in coords])

_bx = land_geom.bounds
bound_minx, bound_miny, bound_maxx, bound_maxy = _bx

minx = bound_minx - 0.3
miny = bound_miny - 0.3
maxx = bound_maxx + 0.3
maxy = min(bound_maxy + 0.3, 14.90)

dlon_deg = (maxx - minx) / 599.0
dlat_deg = (maxy - miny) / 599.0

extent_w = maxx - minx
extent_h = maxy - miny
extent_aspect = extent_w / extent_h

_PANEL_W   = 9.5
_MAP_PAD   = 0.15
_map_w_in  = _PANEL_W - _MAP_PAD * 2
_map_h_in  = _map_w_in / extent_aspect
_PANEL_H   = _map_h_in + _MAP_PAD * 2

_HEADER_IN = 1.35
_FOOTER_IN = 0.85
_CBAR_IN   = 1.00

MAP_FIGSIZE = (_PANEL_W + _CBAR_IN, _PANEL_H + _HEADER_IN + _FOOTER_IN)

gx, gy    = np.linspace(minx, maxx, 600), np.linspace(miny, maxy, 600)
glon, glat = np.meshgrid(gx, gy)

grid_points = np.column_stack((glon.ravel(), glat.ravel()))
mask = np.zeros(len(grid_points), dtype=bool)

for geom in ([land_geom] if not hasattr(land_geom, "geoms") else land_geom.geoms):
    mask |= MplPath(np.array(geom.exterior.coords)).contains_points(grid_points)
    for interior in geom.interiors:
        mask &= ~MplPath(np.array(interior.coords)).contains_points(grid_points)
mask = mask.reshape(glat.shape)

def smooth_interpolation(lons, lats, values, kernel='linear'):
    valid = ~np.isnan(values)
    if valid.sum() < 4:
        return np.full(glat.shape, np.nan)

    lv, lav, vv = lons[valid], lats[valid], values[valid]

    try:
        obs_points = np.column_stack((lv, lav))
        target_points = np.column_stack((glon.ravel(), glat.ravel()))

        rbf = RBFInterpolator(obs_points, vv, kernel=kernel)
        interpolated = rbf(target_points)

        return interpolated.reshape(glat.shape)
    except Exception as e:
        log.warning(f"  RBF interpolation failed: {e}. Returning blank grid.")
        return np.full(glat.shape, np.nan)

# ── 7. STYLING & COLLISION LOGIC ──────────────────────────────────────────────
MCS_TIERS = [
    (0,  1,  (1.0, 1.0, 1.0, 0.0), "0 — Nil"),
    (1,  3,  "#ffffb2", "1–2 — Iso. Conv."),
    (3,  5,  "#fecc5c", "3–4 — Org. Storms"),
    (5,  7,  "#fd8d3c", "5–6 — Squall Line"),
    (7,  9,  "#e31a1c", "7–8 — Severe MCS"),
    (9,  11, "#800026", "9–10 — Extreme"),
]
MCS_COLORS = [c_color for c_low, c_high, c_color, c_label in MCS_TIERS]
MCS_BOUNDARIES = [c_low for c_low, c_high, c_color, c_label in MCS_TIERS]
*_, mcs_last_tier = MCS_TIERS
_, mcs_max, _, _ = mcs_last_tier
MCS_BOUNDARIES.append(mcs_max)
MCS_CMAP  = mcolors.ListedColormap(MCS_COLORS,  name="mcs")
MCS_NORM  = mcolors.BoundaryNorm(MCS_BOUNDARIES, MCS_CMAP.N)

HEAT_TIERS = [
    (0,  2,  (1.0, 1.0, 1.0, 0.0), "0–1 — Normal"),
    (2,  4,  "#ffeda0", "2–3 — Watch"),
    (4,  6,  "#feb24c", "4–5 — Heatwave"),
    (6,  8,  "#f03b20", "6–7 — Severe Heatwave"),
    (8,  11, "#bd0026", "8–10 — Extreme / Baked"),
]
HEAT_COLORS = [h_color for h_low, h_high, h_color, h_label in HEAT_TIERS]
HEAT_BOUNDARIES = [h_low for h_low, h_high, h_color, h_label in HEAT_TIERS]
*_, heat_last_tier = HEAT_TIERS
_, heat_max, _, _ = heat_last_tier
HEAT_BOUNDARIES.append(heat_max)
HEAT_CMAP  = mcolors.ListedColormap(HEAT_COLORS,  name="heat")
HEAT_NORM  = mcolors.BoundaryNorm(HEAT_BOUNDARIES, HEAT_CMAP.N)

ALWAYS_SHOW = {"Gudiyatham", "Vaithiyampettai", "Madurai"}

def get_visible_labels(df, min_deg_sep=0.55):
    shown, keep = [], np.zeros(len(df), dtype=bool)
    for pos, (_, row) in enumerate(df.iterrows()):
        too_close = any(
            np.sqrt((row["Lat"] - s_lat)**2 + (row["Lon"] - s_lon)**2) < min_deg_sep
            for s_lat, s_lon in shown
        )
        if row["City"] in ALWAYS_SHOW or not too_close:
            shown.append((row["Lat"], row["Lon"]))
            keep[pos] = True
    return keep

# ── 8. RENDERERS ──────────────────────────────────────────────────────────────
def get_distance_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))

_TERRAIN_TILER = None  # singleton: shared across all map renders so Cartopy's tile cache is reused

def _add_cartopy_features(ax):
    global _TERRAIN_TILER
    if _TERRAIN_TILER is None:
        _TERRAIN_TILER = GoogleTerrain()
    ax.add_image(_TERRAIN_TILER, 7, zorder=0)

    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#2b2b2b", alpha=0.8, zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor="#2b2b2b", alpha=0.8, zorder=2)

    states = cfeature.NaturalEarthFeature(
        category='cultural', name='admin_1_states_provinces_lines',
        scale='10m', facecolor='none')

    ax.add_feature(states, edgecolor='#888888', linewidth=0.5, alpha=0.6, linestyle=':', zorder=2)
    ax.set_extent([minx, maxx, miny, maxy], crs=ccrs.PlateCarree())

def _fit_footer_lines(parts, fontsize, max_width_in, joiner=" \u00b7 "):
    """Pack ``parts`` onto as many centered lines as needed so each line fits
    within ``max_width_in`` inches. Breaks only at part boundaries (never mid-word),
    and returns the parts unchanged (single line) when they already fit."""
    # DejaVu Sans averages ~0.60 em per glyph; 0.65 adds a safety margin.
    avg_char_in = fontsize * 0.65 / 72.0
    chars_per_line = max(10, int(max_width_in / avg_char_in))
    lines, cur = [], ""
    for part in parts:
        candidate = part if not cur else cur + joiner + part
        if len(candidate) <= chars_per_line:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            cur = part
    if cur:
        lines.append(cur)
    return "\n".join(lines)

def _fit_header_title(product_title, date_str, max_width_in, fontsize=15):
    """Fold the red header title + date onto lines that fit within ``max_width_in``
    inches, breaking at the '  •  ' separator. Shrinks the font as a last resort
    if a single part is still too wide for one line."""
    parts = [product_title, date_str]
    fs = fontsize
    while True:
        text = _fit_footer_lines(parts, fs, max_width_in, joiner="  \u2022  ")
        avg_char_in = fs * 0.65 / 72.0
        longest = max(len(ln) for ln in text.split("\n"))
        if longest * avg_char_in <= max_width_in or fs <= 11:
            return text, fs
        fs -= 0.5

def _add_header_footer(fig, product_title, date_str, top_places_str="", has_cbar=True):
    fig_w = fig.get_figwidth()
    fig_h = fig.get_figheight()
    _hf   = _HEADER_IN / fig_h
    _ff   = _FOOTER_IN / fig_h

    header_bot = 1.0 - _hf
    footer_top = _ff

    # On no-colorbar maps the figure has an empty right gutter (_CBAR_IN wide).
    # Without this correction, ha='right' text at x=0.96 drifts into dead space.
    _txt_right = 0.99 if has_cbar else (_PANEL_W - _MAP_PAD) / fig_w

    fig.add_artist(plt.matplotlib.lines.Line2D([0.0, 1.0], [header_bot, header_bot],
                   transform=fig.transFigure, color="#333333", linewidth=1.5, zorder=10))
    fig.add_artist(plt.matplotlib.lines.Line2D([0.0, 1.0], [footer_top, footer_top],
                   transform=fig.transFigure, color="#333333", linewidth=1.5, zorder=10))

    fig.text(0.04, 1.0 - (_hf * 0.35), "MUGIL ENSEMBLE",
             ha="left", va="center", fontsize=20, color="#111111",
             fontweight="black", fontfamily="DejaVu Sans", zorder=10)

    header_title, header_fs = _fit_header_title(product_title, date_str, _map_w_in)
    fig.text(0.04, 1.0 - (_hf * 0.72), header_title,
             ha="left", va="center", fontsize=header_fs, color="#b22222",
             fontweight="bold", fontfamily="DejaVu Sans", zorder=10)

    fig.text(_txt_right, 1.0 - (_hf * 0.53), MODEL_LABEL_STR.replace(" · ", "  ·  "),
             ha="right", va="center", fontsize=8.5, color="#666666",
             fontweight="normal", fontfamily="DejaVu Sans", zorder=10)

    if top_places_str:
        top_parts = [p.strip() for p in top_places_str.split("  \u00b7  ")]
        fig.text(0.50, footer_top * 0.65,
                 _fit_footer_lines(top_parts, 9.5, _map_w_in),
                 ha="center", va="center", fontsize=9.5, color="#000080",
                 fontweight="bold", fontfamily="DejaVu Sans", zorder=10)

    footer_parts = [
        "\u26a0 Experimental — not for operational aviation, marine or any emergency use",
        "Data: Open-Meteo \u00b7 ECMWF Open Data",
        f"Generated: {RUN_TIMESTAMP}",
    ]
    fig.text(0.50, footer_top * 0.25,
             _fit_footer_lines(footer_parts, 7.5, _map_w_in),
             ha="center", va="center", fontsize=7.5, color="#444444",
             fontstyle="italic", fontfamily="DejaVu Sans", zorder=10)

def save_map(grid, cmap, title, fname, df, col, unit, date_str,
             is_precip=False, norm=None, fixed_vmin=None, fixed_vmax=None,
             tiers=None, is_centered=False):

    fig_w, fig_h = MAP_FIGSIZE
    fig = plt.figure(figsize=MAP_FIGSIZE, facecolor="#ffffff")

    map_left   = _MAP_PAD / fig_w
    map_bottom = (_FOOTER_IN + _MAP_PAD) / fig_h
    map_right  = (_PANEL_W  - _MAP_PAD) / fig_w
    map_top    = (fig_h - _HEADER_IN - _MAP_PAD) / fig_h

    map_w = map_right  - map_left
    map_h = map_top    - map_bottom

    ax = fig.add_axes([map_left, map_bottom, map_w, map_h], projection=ccrs.PlateCarree())

    plot_data = grid.copy().astype(float)
    if is_precip:
        plot_data[plot_data < 1.0] = np.nan
    all_nan = np.all(np.isnan(plot_data))

    if norm is None:
        if is_centered:
            vmax = fixed_vmax if fixed_vmax is not None else float(np.nanmax(np.abs(plot_data)))
            vmin = fixed_vmin if fixed_vmin is not None else -vmax
        else:
            vmin = fixed_vmin if fixed_vmin is not None else (0.0 if all_nan else float(np.nanmin(plot_data)))
            vmax = fixed_vmax if fixed_vmax is not None else (1.0 if all_nan else float(np.nanmax(plot_data)))

        if vmin == vmax:
            vmax = vmin + 1.0

        norm_use = (mcolors.LogNorm(vmin=max(vmin, 1.0), vmax=vmax)
                    if is_precip and not all_nan and vmin >= 1.0
                    else mcolors.Normalize(vmin=vmin, vmax=vmax))
    else:
        norm_use = norm

    im = ax.imshow(
        plot_data,
        extent=[minx, maxx, miny, maxy],
        origin="lower", cmap=cmap,
        alpha=0.65,
        norm=norm_use, transform=ccrs.PlateCarree(),
        zorder=1
    )
    _add_cartopy_features(ax)

    top_places_str = ""
    if col != "dummy" and not df.empty and col in df.columns:
        valid_df = df.dropna(subset=[col])
        top5 = valid_df.sort_values(by=col, ascending=False).head(5)
        top_strs = []
        for _, r in top5.iterrows():
            val = r[col]
            if col in ["Precip", "Precip_P90", "MCS", "HeatScore", "Convergence"] and val <= 0:
                continue
            v_str = f"{val:.1f}{unit}" if col in ["MaxTemp", "MinTemp", "DTR", "ApparentTemp", "ET0", "Convergence"] else f"{int(round(val))}{unit}"
            top_strs.append(f"{r['City']} ({v_str})")

        if top_strs:
            metric_clean_part, *_ = title.split('(')
            metric_clean = metric_clean_part.strip()
            top_places_str = f"Top 5 {metric_clean}:  " + "  ·  ".join(top_strs)

    if not df.empty and col != "dummy":
        keep = get_visible_labels(df, min_deg_sep=0.55)
        for pos, (_, r) in enumerate(df.iterrows()):
            val = r.get(col, np.nan)

            # 1. Skip if the label collision logic says it's too crowded
            if not keep[pos]:
                continue

            # 2. Handle Convergence Map (Print City Name ONLY, clipped to map bounds)
            if col == "Convergence":
                # Skip labels too close to map edges to prevent overflow
                edge_margin = 0.25  # degrees
                if r.Lon < (minx + edge_margin) or r.Lon > (maxx - edge_margin) or \
                   r.Lat < (miny + edge_margin) or r.Lat > (maxy - edge_margin):
                    continue
                txt = ax.text(r.Lon, r.Lat, f"{r.City}",
                               fontsize=7.5, ha="center", weight="bold",
                               zorder=5, transform=ccrs.PlateCarree(),
                               clip_on=True)
                txt.set_path_effects([pe.withStroke(linewidth=2.8, foreground="white")])
                continue

            # 3. Handle All Other Maps (Must have valid data)
            if np.isnan(val):
                continue
            if norm is not None and round(val) == 0:
                continue

            val_str = f"{val:.1f}{unit}" if col in ["MaxTemp", "MinTemp", "DTR", "ApparentTemp", "ET0"] else f"{int(round(val))}{unit}"
            txt = ax.text(
                r.Lon, r.Lat,
                f"{r.City}\n{val_str}",
                fontsize=8.5, ha="center", weight="bold",
                zorder=5, transform=ccrs.PlateCarree()
            )
            txt.set_path_effects([pe.withStroke(linewidth=2.8, foreground="white")])

    if norm is not None:
        _tiers = tiers if tiers is not None else MCS_TIERS
        legend_handles = [mpatches.Patch(color=t_color, label=t_label) for t_low, t_high, t_color, t_label in _tiers]
        ax.legend(handles=legend_handles, loc="lower right", fontsize=8.5,
                  title=title, title_fontsize=9, framealpha=0.85)
    else:
        _cbar_gap   = 0.08
        _cbar_bar_w = 0.22
        _cbar_l_in  = _PANEL_W + _cbar_gap
        _cbar_b_in  = _FOOTER_IN + _MAP_PAD * 2
        _cbar_h_in  = _PANEL_H - _MAP_PAD * 4

        cbar_ax = fig.add_axes([
            _cbar_l_in  / fig_w,
            _cbar_b_in  / fig_h,
            _cbar_bar_w / fig_w,
            _cbar_h_in  / fig_h,
        ])

        cb = fig.colorbar(im, cax=cbar_ax, extend="both" if is_centered else ("max" if is_precip else "neither"))

        sf = ScalarFormatter()
        sf.set_scientific(False)
        cb.ax.yaxis.set_major_formatter(sf)
        cb.ax.yaxis.set_minor_formatter(sf)

        cb.set_label(title, fontsize=8.5, labelpad=5, fontfamily="DejaVu Sans")
        cb.ax.tick_params(labelsize=8)

    _add_header_footer(fig, title, date_str, top_places_str)

    plt.savefig(os.path.join(OUT_DIR, fname), dpi=200, bbox_inches=None, facecolor="#ffffff")
    plt.close(fig)

def save_synoptic_map(grid, title, fname, date_str):
    fig_w, fig_h = MAP_FIGSIZE
    fig = plt.figure(figsize=MAP_FIGSIZE, facecolor="#ffffff")

    map_left, map_bottom = _MAP_PAD / fig_w, (_FOOTER_IN + _MAP_PAD) / fig_h
    map_w, map_h = (_PANEL_W - _MAP_PAD*2) / fig_w, (fig_h - _HEADER_IN - _FOOTER_IN - _MAP_PAD*2) / fig_h

    ax = fig.add_axes([map_left, map_bottom, map_w, map_h], projection=ccrs.PlateCarree())

    masked_mslp = np.where(gelev > 1000, np.nan, grid)

    levels = np.arange(980, 1025, 2)
    contours = ax.contour(glon, glat, masked_mslp, levels=levels, colors="black",
                          linewidths=1.2, alpha=0.8, transform=ccrs.PlateCarree(), zorder=1)

    ax.clabel(contours, inline=True, fontsize=8, fmt="%1.0f")

    # Increased filter size to 150 (approx 180 km) to capture true synoptic-scale LPAs
    neighborhood = 150
    local_min = ndimage.minimum_filter(masked_mslp, size=neighborhood) == masked_mslp
    is_lpa = local_min & (masked_mslp < 1010)

    min_indices = np.argwhere(is_lpa)
    accepted = []

    for y, x in min_indices:
        lat, lon = glat[y, x], glon[y, x]

        too_close = False
        for a_lat, a_lon, a_val in accepted:
            dist_km = get_distance_km(lat, lon, a_lat, a_lon)
            if dist_km <= 160:
                too_close = True
                break

        if not too_close:
            accepted.append((lat, lon, masked_mslp[y, x]))

    for lat, lon, val in accepted:
        ax.text(lon, lat, 'L', color='#b22222', fontsize=22, weight='black',
                ha='center', va='center', zorder=10, transform=ccrs.PlateCarree(),
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        ax.text(lon, lat-0.2, f"{int(val)} hPa", color='#b22222', fontsize=9, weight='bold',
                ha='center', va='center', zorder=10, transform=ccrs.PlateCarree(),
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    _add_cartopy_features(ax)
    _add_header_footer(fig, title, date_str, "Synoptic Mean Sea Level Pressure (MSLP) & Low Pressure Areas", has_cbar=False)

    plt.savefig(os.path.join(OUT_DIR, fname), dpi=200, bbox_inches=None, facecolor="#ffffff")
    plt.close(fig)

# [MUGIL v19.3 PATCH] New Contoured Convergence Map Renderer
def save_convergence_contour_map(grid, title, fname, df, date_str):
    fig_w, fig_h = MAP_FIGSIZE
    fig = plt.figure(figsize=MAP_FIGSIZE, facecolor="#ffffff")

    map_left, map_bottom = _MAP_PAD / fig_w, (_FOOTER_IN + _MAP_PAD) / fig_h
    map_w, map_h = (_PANEL_W - _MAP_PAD*2) / fig_w, (fig_h - _HEADER_IN - _FOOTER_IN - _MAP_PAD*2) / fig_h

    ax = fig.add_axes([map_left, map_bottom, map_w, map_h], projection=ccrs.PlateCarree())

    # Mask offshore data so contours cleanly terminate at the coast
    plot_data = np.where(mask, grid, np.nan)

    # Plot Contours for Convergence (Positive values indicating lift, in 10^-5 s^-1)
    levels = [1.5, 3.0, 5.0, 8.0, 12.0]
    linewidths = [1.5, 2.0, 2.5, 3.0, 3.5]
    colors = ["#74c476", "#31a354", "#006d2c", "#756bb1", "#54278f"] # Greens progressing to deep Purples

    contours = ax.contour(glon, glat, plot_data, levels=levels, colors=colors,
                          linewidths=linewidths, alpha=0.9, transform=ccrs.PlateCarree(), zorder=3)

    # Add inline labels to the contour lines
    clabels = ax.clabel(contours, inline=True, fontsize=9, fmt="%1.1f")
    for txt in clabels:
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])

    _add_cartopy_features(ax)

    # Add Places Labels
    if not df.empty:
        keep = get_visible_labels(df, min_deg_sep=0.55)
        edge_margin = 0.25  # degrees — skip labels near map edges to prevent overflow
        for pos, (_, r) in enumerate(df.iterrows()):
            if not keep[pos]:
                continue
            # Skip labels too close to map edges
            if r.Lon < (minx + edge_margin) or r.Lon > (maxx - edge_margin) or \
               r.Lat < (miny + edge_margin) or r.Lat > (maxy - edge_margin):
                continue
            # Draw city dot
            ax.plot(r.Lon, r.Lat, marker='o', color='#333333', markersize=4.0, transform=ccrs.PlateCarree(), zorder=5)
            # Draw city name directly below the dot
            txt = ax.text(
                r.Lon, r.Lat - 0.08,
                f"{r.City}",
                fontsize=7.5, ha="center", va="top", weight="bold", color="#111111",
                zorder=6, transform=ccrs.PlateCarree(), clip_on=True
            )
            txt.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white")])

    # Custom Legend for the contour lines
    legend_elements = [
        plt.matplotlib.lines.Line2D((0,), (0,), color="#31a354", lw=2.0, label='Mod. Conv (>3.0 × 10⁻⁵ s⁻¹)'),
        plt.matplotlib.lines.Line2D((0,), (0,), color="#54278f", lw=3.5, label='Str. Conv (>12.0 × 10⁻⁵ s⁻¹)')
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8.5, framealpha=0.9)

    _add_header_footer(fig, title, date_str, "Synoptic Wind Convergence (Solid Lines = Lift / Storm Initiation)", has_cbar=False)

    plt.savefig(os.path.join(OUT_DIR, fname), dpi=200, bbox_inches=None, facecolor="#ffffff")
    plt.close(fig)

def save_google_terrain_basemap(fname, date_str):
    fig_w, fig_h = MAP_FIGSIZE
    fig = plt.figure(figsize=MAP_FIGSIZE, facecolor="#ffffff")

    map_left   = _MAP_PAD / fig_w
    map_bottom = (_FOOTER_IN + _MAP_PAD) / fig_h
    map_right  = (_PANEL_W  - _MAP_PAD) / fig_w
    map_top    = (fig_h - _HEADER_IN - _MAP_PAD) / fig_h

    map_w = map_right  - map_left
    map_h = map_top    - map_bottom

    ax = fig.add_axes([map_left, map_bottom, map_w, map_h], projection=ccrs.PlateCarree())

    _add_cartopy_features(ax)
    _add_header_footer(fig, "Reference Map: Google Terrain", date_str, has_cbar=False)

    plt.savefig(os.path.join(OUT_DIR, fname), dpi=200, bbox_inches=None, facecolor="#ffffff")
    plt.close(fig)

# ── 9. GRIB AUTOMATION (ECMWF Open Data) ──────────────────────────────────────
def get_ecmwf_grib_url():
    now = datetime.now(timezone.utc)
    if now.hour < 10:
        run_time = now - timedelta(days=1)
        cycle = "12"
        step = 24  # 12z + 24h = 12z today
    elif now.hour < 21:
        run_time = now
        cycle = "00"
        step = 12  # 00z + 12h = 12z today
    else:
        run_time = now
        cycle = "12"
        step = 24  # 12z + 24h = 12z tomorrow

    date_str = run_time.strftime("%Y%m%d")
    url = f"https://data.ecmwf.int/forecasts/{date_str}/{cycle}z/ifs/0p25/oper/{date_str}{cycle}0000-{step}h-oper-fc.grib2"
    return url, cycle, step

def _try_install_cfgrib():
    import subprocess, sys
    log.info("  cfgrib not found — attempting auto-install (requires libeccodes)...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "cfgrib", "--break-system-packages", "-q"],
            capture_output=True, timeout=60
        )
        if result.returncode == 0:
            import cfgrib  # noqa: F401
            log.info("  ✓ cfgrib installed successfully.")
            return True
        else:
            log.warning(f"  pip install cfgrib failed:\n{result.stderr.decode().strip()}")
    except Exception as e:
        log.warning(f"  Auto-install failed: {e}")
    return False

def process_grib_file(grib_path, var_name="2t", apply_mask=True):
    global _CFGRIB_OK
    if not _CFGRIB_OK:
        _CFGRIB_OK = _try_install_cfgrib()
    if not _CFGRIB_OK:
        log.error("  ✘ cfgrib engine unavailable.")
        return None

    log.info(f"  Reading ECMWF GRIB: {grib_path}")
    try:
        import cfgrib
        datasets = cfgrib.open_datasets(grib_path)

        if var_name in ["msl", "prmsl", "PRMSL", "mslet"]:
            candidates = ["msl", "prmsl", "PRMSL", "mslet"]
        elif var_name in ["2t", "t2m", "TMP", "t"]:
            candidates = ["2t", "t2m", "TMP", "t"]
        elif var_name in ["10u", "u10", "10si", "u"]:
            candidates = ["10u", "u10", "10si", "u"]
        elif var_name in ["10v", "v10", "10si", "v"]:
            candidates = ["10v", "v10", "10si", "v"]
        else:
            candidates = [var_name]

        target_ds = None
        matched = None

        for ds in datasets:
            matched = next((v for v in candidates if v in ds.data_vars), None)
            if matched:
                target_ds = ds
                break

        if target_ds is None:
            log.warning(f"  Target variable '{var_name}' not found in any GRIB hypercube.")
            return None

        data = target_ds[matched]

        if float(data.longitude.max()) > 180:
            data = data.assign_coords(longitude=(((data.longitude + 180) % 360) - 180))
            data = data.sortby("longitude")

        interpolated = data.interp(longitude=gx, latitude=gy, method="linear")
        grid_vals = interpolated.values

        if matched in ["msl", "prmsl", "PRMSL", "mslet"]:
            grid_vals = grid_vals / 100.0

        if apply_mask:
            return np.where(mask, grid_vals, np.nan)

        return grid_vals

    except Exception as e:
        log.error(f"  Failed to read GRIB file: {e}")
        return None

# ── 10. MAIN EXECUTION LOOP ───────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("📡 Step 1: Fetching Open-Meteo Ensemble & API Data (Async + Cache).")

    elevs      = get_elevations(coords)
    daily_rows = {dt.strftime("%Y-%m-%d"): [] for dt in forecast_days}

    all_city_data = asyncio.run(fetch_all_cities_async(locations))

    for i, (name, (lat, lon)) in enumerate(locations.items()):
        city_data = all_city_data.get(name, {})

        if not city_data:
            log.warning(f"  Empty API response for {name}")
            continue

        for day_idx, dt in enumerate(forecast_days):
            dt_str     = dt.strftime("%Y-%m-%d")
            model_data = city_data.get(dt_str, {})
            if not model_data:
                continue
            try:
                stats = compute_ensemble_stats(model_data, day_idx)

                tmax_val = stats.get("temperature_2m_max", np.nan)
                tmin_val = stats.get("temperature_2m_min", np.nan)
                dp_val   = stats.get("dewpoint_2m_mean", np.nan)

                dtr = (tmax_val - tmin_val) if not (np.isnan(tmax_val) or np.isnan(tmin_val)) else np.nan
                elev_km = elevs[i] / 1000.0

                daily_rows[dt_str].append({
                    "City":     name,
                    "Lat":      lat,
                    "Lon":      lon,
                    "StationElev": elev_km,
                    "MaxTemp":  tmax_val,
                    "MinTemp":  tmin_val,
                    "DTR":      dtr,
                    "Precip":   stats.get("precipitation_sum", np.nan),
                    "Precip_P90": stats.get("precip_p90", np.nan),
                    "Precip_Max": stats.get("precip_max", np.nan),
                    "Gusts":    stats.get("windgusts_10m_max", np.nan),
                    "WindDir":  stats.get("winddirection_10m_dominant", np.nan),
                    "U_wind":   stats.get("u_wind", np.nan),
                    "V_wind":   stats.get("v_wind", np.nan),
                    "CAPE":     stats.get("cape_mean", np.nan),
                    "Dewpoint": dp_val,
                    "VPD":      stats.get("vapour_pressure_deficit_max", np.nan),
                    "ET0":      stats.get("et0_fao_evapotranspiration", np.nan),
                    "PrecipProb": stats.get("precip_prob", np.nan),
                    "MCS":      compute_mcs_score(stats),
                    "HeatScore": compute_heat_score(stats),
                    "ApparentTemp": compute_heat_index(tmax_val, dp_val),
                })
            except Exception as exc:
                log.error(f"  Math/Stat error for {name} on {dt_str}: {exc}")

    topo_lats  = np.array([t_lat for t_lat, t_lon, t_elev in topo_anchors])
    topo_lons  = np.array([t_lon for t_lat, t_lon, t_elev in topo_anchors])
    topo_elevs = np.array([t_elev for t_lat, t_lon, t_elev in topo_anchors], dtype=float)

    all_lons  = np.concatenate([lons_arr, topo_lons])
    all_lats  = np.concatenate([lats_arr, topo_lats])
    all_elevs = np.concatenate([np.array(elevs, dtype=float), topo_elevs])

    gelev = np.clip(smooth_interpolation(all_lons, all_lats, all_elevs, kernel='linear'), 0, None)

    log.info("⛰️ Rendering clean Google Terrain Basemap...")
    first_date_str = next(iter(forecast_days)).strftime("%d %b %Y")
    save_google_terrain_basemap("00_google_terrain_basemap.png", first_date_str)

    log.info("🎨 Step 2: Rendering Daily Maps...")
    for day_idx, target_dt in enumerate(forecast_days):
        dt_str     = target_dt.strftime("%Y-%m-%d")
        date_label = target_dt.strftime("%d %b %Y")
        df = pd.DataFrame(daily_rows.get(dt_str, []))
        if df.empty:
            continue

        tag      = f"T{day_idx+1}_{dt_str}"
        lons_d   = df.Lon.values
        lats_d   = df.Lat.values

        # Dynamic lapse rate for daytime temperatures
        dynamic_lapse = np.clip(5.0 + (df.DTR.values - 5) * (4.8 / 5.0), 5.0, 9.8)
        z_lapse = smooth_interpolation(lons_d, lats_d, dynamic_lapse, kernel='linear')

        # Sea-level reduction downscaling for MaxTemp
        tmax_sl = df.MaxTemp.values + df.StationElev.values * dynamic_lapse
        z_tmax_sl = smooth_interpolation(lons_d, lats_d, tmax_sl, kernel='linear')
        grid_tmax = np.where(mask, z_tmax_sl - (gelev / 1000.0) * z_lapse, np.nan)
        
        save_map(grid_tmax, "magma_r", "Peak Temperature (°C)", f"01_maxtemp_{tag}.png",
                 df, "MaxTemp", "°C", date_label, fixed_vmin=24, fixed_vmax=44)

        z_precip    = smooth_interpolation(lons_d, lats_d, df.Precip_P90.values, kernel='linear')
        grid_precip = np.where(mask, z_precip, np.nan)
        save_map(grid_precip, "viridis", "Precipitation 90th Pctl (mm/day)", f"02_precip_{tag}.png",
                 df, "Precip_P90", "mm/day", date_label, is_precip=True)

        z_mcs    = smooth_interpolation(lons_d, lats_d, df.MCS.values.astype(float), kernel='linear')
        grid_mcs = np.where(mask, z_mcs, np.nan)
        save_map(grid_mcs, MCS_CMAP, "MCS / Thundersquall Index", f"03_mcs_{tag}.png",
                 df, "MCS", "", date_label, norm=MCS_NORM, tiers=MCS_TIERS)

        z_gusts    = smooth_interpolation(lons_d, lats_d, df.Gusts.values, kernel='linear')
        grid_gusts = np.where(mask, z_gusts, np.nan)
        save_map(grid_gusts, "plasma_r", "Wind Gusts (km/h)", f"04_gusts_{tag}.png",
                 df, "Gusts", "km/h", date_label, fixed_vmin=0, fixed_vmax=80)

        z_apptemp = smooth_interpolation(lons_d, lats_d, df.ApparentTemp.values, kernel='linear')
        grid_apptemp = np.where(mask, z_apptemp, np.nan)
        save_map(grid_apptemp, "OrRd", "Apparent Temp / Heat Index (°C)", f"05a_heat_index_{tag}.png",
                 df, "ApparentTemp", "°C", date_label, fixed_vmin=30, fixed_vmax=50)

        z_heat = smooth_interpolation(lons_d, lats_d, df.HeatScore.values.astype(float), kernel='linear')
        grid_heat = np.where(mask, z_heat, np.nan)
        save_map(grid_heat, HEAT_CMAP, "Heatwave Alert Index", f"05b_heatwave_{tag}.png",
                 df, "HeatScore", "", date_label, norm=HEAT_NORM, tiers=HEAT_TIERS)

        z_dtr = smooth_interpolation(lons_d, lats_d, df.DTR.values, kernel='linear')
        grid_dtr = np.where(mask, z_dtr, np.nan)
        save_map(grid_dtr, "turbo", "Diurnal Temp Range (°C)", f"06_dtr_{tag}.png",
                 df, "DTR", "°C", date_label, fixed_vmin=2, fixed_vmax=18)

        # Sea-level reduction downscaling for MinTemp using a separate constant nighttime lapse rate (3.0°C/km)
        tmin_sl = df.MinTemp.values + df.StationElev.values * 3.0
        z_tmin_sl = smooth_interpolation(lons_d, lats_d, tmin_sl, kernel='linear')
        grid_tmin = np.where(mask, z_tmin_sl - (gelev / 1000.0) * 3.0, np.nan)
        
        save_map(grid_tmin, "inferno", "Nighttime Heat Stress / Min Temp (°C)", f"07_mintemp_{tag}.png",
                 df, "MinTemp", "°C", date_label, fixed_vmin=18, fixed_vmax=32)

        z_et0 = smooth_interpolation(lons_d, lats_d, df.ET0.values, kernel='linear')
        grid_et0 = np.where(mask, z_et0, np.nan)
        save_map(grid_et0, "YlOrBr", "Evapotranspiration ET0 (mm/day)", f"08_et0_{tag}.png",
                 df, "ET0", "mm", date_label, fixed_vmin=2, fixed_vmax=8)

        # Daily convergence: converted to m/s, physically scaled to meters, and presented in 10^-5 s^-1
        try:
            z_u = smooth_interpolation(lons_d, lats_d, df.U_wind.values, kernel='thin_plate_spline')
            z_v = smooth_interpolation(lons_d, lats_d, df.V_wind.values, kernel='thin_plate_spline')

            # Convert wind from km/h to m/s
            z_u_ms = z_u / 3.6
            z_v_ms = z_v / 3.6

            # Compute physical coordinate spacing in meters
            lat_rad = np.radians(glat)
            dx = dlon_deg * 111139.0 * np.cos(lat_rad)
            dy = dlat_deg * 111139.0

            du_dy_idx, du_dx_idx = np.gradient(z_u_ms)
            dv_dy_idx, dv_dx_idx = np.gradient(z_v_ms)

            du_dx = du_dx_idx / dx
            dv_dy = dv_dy_idx / dy

            raw_convergence = -(du_dx + dv_dy)
            # Scale to standard meteorological units (10^-5 s^-1)
            raw_convergence_scaled = raw_convergence * 100000.0
            
            smoothed_conv = ndimage.gaussian_filter(raw_convergence_scaled, sigma=8)
            grid_conv = np.where(mask, smoothed_conv, np.nan)

            df["Convergence"] = np.nan
            save_map(grid_conv, "PiYG", "Wind Convergence (×10⁻⁵ s⁻¹)  [+ve = Convergence / Lift]", f"09_convergence_{tag}.png",
                     df, "Convergence", "", date_label, fixed_vmin=-5.0, fixed_vmax=5.0, is_centered=True)
        except Exception as e:
            log.warning(f"  Failed to render daily convergence for day {day_idx+1}: {e}")

        log.info(f"  ✓ Day {day_idx+1} ({dt_str}) — maps saved.")

    log.info("\n🌪️  Step 3: Triggering GRIB Automation (ECMWF Open Data)...")
    first_day_str = next(iter(forecast_days)).strftime("%Y-%m-%d")
    df_today = pd.DataFrame(daily_rows.get(first_day_str, []))

    ecmwf_url, cycle, step = get_ecmwf_grib_url()
    temp_grib_path = f"temp_ecmwf_{int(time.time())}.grib2"

    try:
        log.info(f"  📥 Downloading ECMWF Open Data GRIB ({cycle}z cycle, +{step}h) once for all maps...")
        with SESSION.get(ecmwf_url, stream=True, timeout=120) as r:
            r.raise_for_status()
            content_type = r.headers.get("Content-Type", "")
            if "html" in content_type.lower():
                raise ValueError("ECMWF returned HTML. Cycle not published yet or URL is invalid.")
            with open(temp_grib_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)

        grid_t2m = process_grib_file(temp_grib_path, "2t")
        if grid_t2m is not None:
            if np.nanmax(grid_t2m) > 200: grid_t2m -= 273.15
            grib_date_label = datetime.now().strftime("%d %b %Y") + f"  (ECMWF {cycle}z +{step}h Snapshot)"
            save_map(grid_t2m, "magma_r", "ECMWF Raw Temperature Snapshot (°C)", "00_ecmwf_raw_t2m.png",
                     df_today, "MaxTemp", "°C", grib_date_label)

        grid_mslp = process_grib_file(temp_grib_path, "msl", apply_mask=False)
        if grid_mslp is not None:
            grib_date_label = datetime.now().strftime("%d %b %Y") + f"  (ECMWF {cycle}z +{step}h Snapshot)"
            save_synoptic_map(grid_mslp, "Mean Sea Level Pressure (hPa)", "00_ecmwf_mslp_lpa.png", grib_date_label)

        # [MUGIL v19.3 PATCH] True ECMWF High-Res Convergence Layer (Contoured & Physically Scaled)
        grid_u10 = process_grib_file(temp_grib_path, "u10")
        grid_v10 = process_grib_file(temp_grib_path, "v10")

        if grid_u10 is not None and grid_v10 is not None:
            log.info("  ✓ Calculating high-resolution ECMWF convergence...")
            
            # Compute physical coordinate spacing in meters
            lat_rad = np.radians(glat)
            dx = dlon_deg * 111139.0 * np.cos(lat_rad)
            dy = dlat_deg * 111139.0

            du_dy_idx, du_dx_idx = np.gradient(grid_u10)
            dv_dy_idx, dv_dx_idx = np.gradient(grid_v10)

            du_dx = du_dx_idx / dx
            dv_dy = dv_dy_idx / dy

            ecmwf_conv = -(du_dx + dv_dy)
            # Scale to standard meteorological units (10^-5 s^-1)
            ecmwf_conv_scaled = ecmwf_conv * 100000.0

            # Increased smoothing to 2.0 to make contour lines flow smoothly across the 0.25 deg grid
            ecmwf_conv_smooth = ndimage.gaussian_filter(ecmwf_conv_scaled, sigma=2.0)

            grib_date_label = datetime.now().strftime("%d %b %Y") + f"  (ECMWF {cycle}z +{step}h Snapshot)"

            # Execute new contour renderer
            save_convergence_contour_map(ecmwf_conv_smooth, "ECMWF True Grid Wind Convergence (10⁻⁵ s⁻¹)",
                                         "00_ecmwf_convergence_snapshot.png", df_today, grib_date_label)

    except Exception as e:
        log.warning(f"  ✘ ECMWF GRIB automation failed or bypassed: {e}")
    finally:
        if os.path.exists(temp_grib_path):
            os.remove(temp_grib_path)

    log.info("\n🌊 Step 4: Generating Oceanic Diagnostics (IOD & Pacific SST for El Niño/La Niña)...")
    try:
        from mugil_sst_maps import generate_ocean_sst_maps
        generate_ocean_sst_maps(OUT_DIR, RUN_TIMESTAMP)
    except Exception as e:
        log.warning(f"  ✘ Ocean SST map generation failed or bypassed: {e}")

    log.info(f"\n✅ Pipeline complete → '{OUT_DIR}/'")
