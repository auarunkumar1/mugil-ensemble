# MUGIL Ensemble Weather Forecasting Pipeline — v19.5
# ========================================================
# Generates interpolated meteorological maps for South India from
# multiple NWP and AI models via Open-Meteo & ECMWF Open Data.
#
# v19.5 Upgrades — ensemble trueness & physical validity:
#   1. BATCHED FETCH: one multi-location POST per model (5-11 HTTP calls total
#      instead of 390). v19.4's 78x5 request storm tripped Open-Meteo's hourly
#      limit, and its 0.5/1/2 s retry could never satisfy a ~60 s 429 backoff —
#      so AIFS and WeatherNext silently vanished at 58 of 78 stations.
#   2. TRUE MEMBER STATISTICS: P90 / probability-of-rain / spread are computed
#      from the ensemble members (30-63 per model) instead of from 3-5
#      deterministic model values. v19.4's "P90" was the 2nd-highest of five
#      models and its "probability" a 5-sample vote.
#   3. HOMOGENEOUS MODEL SET: every (day, variable) is blended over ONE model
#      set used at every station feeding the interpolator, so the RBF no longer
#      mixes two different estimators across the domain.
#   4. PER-VARIABLE MODEL SETS: a variable is blended only over the models that
#      actually provide it (gusts/CAPE = IFS+ICON+GFS; dewpoint has no WN2).
#   5. HOURLY WIND VECTORS: daily-mean u/v from hourly wind replaces the
#      windspeed_max x 0.50 heuristic, and components (not angles) are averaged.
#   6. CONVERGENCE RECALIBRATION: the daily map is auto-scaled to its own p95
#      (v19.4 used +-5 against a p99 of 2.7 -> a blank map); the offshore TPS
#      extrapolation is tapered out beyond the station confidence radius; the
#      ECMWF contour levels match the reachable 0.25 deg distribution.
#   7. GRIB NaN HYGIENE: gradients are taken on the unmasked field and the mask
#      is applied afterwards (v19.4 was NaN-poisoning the coastal band), and the
#      LPA neighbourhood is derived from the real grid geometry (~1.3 km/px).
#   8. CACHE v2: schema-validated, NaN-free JSON. Degraded runs are not cached,
#      so a rate-limited day can never be replayed as if it were complete.
#
# v19.4.2 Upgrades:
#   1. NOAA OISST NEAR-REAL-TIME (NRT) ROUTING: Switched ERDDAP dataset targets
#      to ncdcOisst21NrtAgg_LonPM180 and ncdcOisst21NrtAgg with delayed-mode fallback,
#      eliminating the ~16-day latency on oceanic SST maps.
#
# v19.4 Upgrades:
#   0. TODO (v20): Upgrade google_weathernext2_ensemble to google_weathernext3_ensemble
#      when WeatherNext 3 becomes available on Open-Meteo ensemble endpoint.
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
import re
from collections import Counter
import numpy as np
import pandas as pd
import requests
import time
import os
import json
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
except Exception as exc:  # optional shim — never fatal, but never silent
    import sys as _sys
    _sys.stderr.write(f"  findlibs/ecCodes shim skipped: {exc}\n")

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
# TODO (v20): Swap google_weathernext2_ensemble -> google_weathernext3_ensemble when available
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
STAGING_CACHE_FILE = f"api_cache_staging_{RUN_DATE_STR}.json"
CACHE_SCHEMA_VERSION = "19.5"
os.makedirs(OUT_DIR, exist_ok=True)

forecast_days = [datetime.now() + timedelta(days=d) for d in range(0, 8)]

ENSEMBLE_MODELS = {
    "ecmwf_ifs025":            "ECMWF IFS",
    "ecmwf_aifs025":           "ECMWF AIFS",
    "google_weathernext2_ensemble": "WeatherNext 2",
    "icon_global":             "ICON Global",
    "gfs_seamless":            "GFS",
}
# Rewritten at runtime from what was actually blended (see set_model_label).
MODEL_LABEL_STR = " · ".join(ENSEMBLE_MODELS.values())
RUN_NOTES = []   # degraded-run disclosures appended to every footer


def note_once(msg):
    if msg not in RUN_NOTES:
        RUN_NOTES.append(msg)
        log.warning(f"  ⚠ {msg}")

DAILY_VARS = [
    "temperature_2m_max", "temperature_2m_min", "windspeed_10m_max",
    "windgusts_10m_max", "precipitation_sum", "cape_mean",
    "dewpoint_2m_mean", "vapour_pressure_deficit_max",
    "winddirection_10m_dominant", "et0_fao_evapotranspiration"
]

# Per-model daily variables supported on the ensemble endpoint.
# Restricting queries to variables a model actually populates slashes Open-Meteo
# request weight by ~40%, preventing 429 quota exhaustion.
MODEL_ENSEMBLE_VARS = {
    "ecmwf_ifs025": DAILY_VARS,
    "ecmwf_aifs025": [v for v in DAILY_VARS if v not in ("windgusts_10m_max", "cape_mean")],
    "google_weathernext2_ensemble": [
        "temperature_2m_max", "temperature_2m_min", "windspeed_10m_max",
        "precipitation_sum", "winddirection_10m_dominant"
    ],
    "icon_global": DAILY_VARS,
    "gfs_seamless": DAILY_VARS,
}

# Variables whose full member distribution is cached (pooled percentile / probability).
# Everything else keeps mean + p10 + p90 + n, which is all the blend and spread need.
FULL_MEMBER_VARS = ("precipitation_sum",)
# Threshold (mm/day) used for the ensemble probability-of-rain product.
PRECIP_PROB_THRESHOLD_MM = 1.0

ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
# Models with no deterministic run on the forecast endpoint must come from the
# ensemble endpoint. Predicate is explicit (v19.4 used a substring test that
# would also catch e.g. a hypothetical "ecmwf_aifs_single").
ENSEMBLE_ONLY_MODELS = {"ecmwf_aifs025", "google_weathernext2_ensemble"}

# Models that expose a cheap hourly wind series on the *deterministic* endpoint.
# AIFS/WN2 only serve hourly through the ensemble endpoint, where the 50-63
# members multiply the payload ~50x for a field that needs one vector per day.
# They are therefore excluded from the wind field (a legitimate per-variable
# model set, like gusts/CAPE) instead of forcing 15-20 MB responses per model.
WIND_MODELS = ("ecmwf_ifs025", "icon_global", "gfs_seamless")

_MEMBER_RE = re.compile(r"^(?P<var>.+)_member(?P<idx>\d+)$")


def set_model_label(models_used, member_counts, det_only=()):
    """Reflect the models actually blended in this run, with real member counts."""
    global MODEL_LABEL_STR
    parts = []
    for mk in ENSEMBLE_MODELS:
        if mk not in models_used:
            continue
        n = member_counts.get(mk, 0)
        tag = f"{n} mem" if n and mk not in det_only else "det"
        parts.append(f"{ENSEMBLE_MODELS[mk]} ({tag})")
    MODEL_LABEL_STR = " · ".join(parts) if parts else "no model data"

# ── 2. HTTP SESSION ───────────────────────────────────────────────────────────
def make_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "MUGIL-Ensemble-Pipeline/19.5 (research; non-commercial)"})
    retry_codes = list((500, 502, 503, 504))   # 429 is handled explicitly (needs ~60 s waits)
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

# ── 4. BATCHED FORECAST FETCHING & CACHING ────────────────────────────────────
# Open-Meteo accepts many coordinate pairs per request, so the whole domain is
# fetched with 5-15 multi-location calls instead of v19.4's 390 single-location
# calls.  That storm exhausted the ensemble endpoint's hourly budget, and the
# 0.5/1/2 s retry could never outlast an hourly limit: AIFS and WeatherNext
# silently disappeared from 58 of 78 stations, leaving the RBF interpolator
# mixing a 5-model estimator and a 3-model estimator across the same map.

_LAST_CALL = [0.0]


def _pace(min_gap=5.0):
    """Fair-use pacing: keep at least ``min_gap`` seconds between API calls."""
    gap = time.time() - _LAST_CALL[0]
    if gap < min_gap:
        time.sleep(min_gap - gap)
    _LAST_CALL[0] = time.time()


def _post_json(url, payload, label, tries=4, first_wait=30):
    """POST with a 429-aware backoff distinguishing minutely vs hourly limits."""
    wait = first_wait
    for _attempt in range(tries):
        _pace()
        try:
            r = SESSION.post(url, data=payload, timeout=180)
        except Exception as exc:
            log.warning(f"    {label}: network error ({exc}); retrying in {wait}s")
            time.sleep(wait)
            wait = min(wait * 2, 240)
            continue
        if r.status_code == 429:
            reason = ""
            try:
                reason = r.json().get("reason", "")
            except Exception:
                reason = r.text[:120]
            if "Hourly" in reason or "Daily" in reason:
                log.error(f"    {label}: HTTP 429 ({reason}) — stopping retries until next hour")
                return None
            sleep_sec = 60 if "Minutely" in reason else wait
            log.warning(f"    {label}: HTTP 429 ({reason or 'rate limit'}); retrying in {sleep_sec}s")
            time.sleep(sleep_sec)
            wait = min(wait * 2, 240)
            continue
        if r.status_code != 200:
            log.warning(f"    {label}: HTTP {r.status_code} — {r.text[:160]}")
            return None
        try:
            return r.json()
        except Exception as exc:
            log.warning(f"    {label}: unreadable JSON ({exc})")
            return None
    log.error(f"    {label}: gave up after {tries} attempts (rate limited)")
    return None


def _multi_payload(coords, days=8, **extra):
    """Request body for a multi-location call. Order is preserved by index."""
    payload = {
        "latitude": ",".join(f"{lat_val:.4f}" for lat_val, _ in coords),
        "longitude": ",".join(f"{lon_val:.4f}" for _, lon_val in coords),
        "timezone": "auto",
        "forecast_days": days,
    }
    payload.update(extra)
    return payload


def _as_locations(js, expected, label):
    """Normalise a 1-or-N location response. Never silently misalign."""
    if js is None:
        return None
    if isinstance(js, dict) and js.get("error"):
        log.warning(f"    {label}: API error — {js.get('reason', js)}")
        return None
    arr = js if isinstance(js, list) else [js]
    if len(arr) != expected:
        log.error(f"    {label}: expected {expected} locations, got {len(arr)} — skipping")
        return None
    return arr


def _stat_from_values(vals):
    """Mean / p10 / p90 / member-count from one model's members for one day."""
    if not vals:
        return None
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return {
        "mean": float(arr.mean()),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "n": int(arr.size),
    }


def _daily_stats(daily):
    """{date: {var: stat}} from one location's daily block (members included)."""
    times = daily.get("time") or []
    member_keys = {}
    for key in daily:
        m = _MEMBER_RE.match(key)
        if m:
            member_keys.setdefault(m.group("var"), []).append((int(m.group("idx")), key))
    for keys in member_keys.values():
        keys.sort()

    out = {}
    for i, dt in enumerate(times):
        row = {}
        for var in DAILY_VARS:
            vals = []
            for _idx, key in member_keys.get(var, ()):
                col = daily.get(key)
                if col is None or i >= len(col):
                    continue
                v = col[i]
                if v is not None:
                    vals.append(float(v))

            stat = _stat_from_values(vals)
            if stat is None:
                # No members (deterministic-only model, or a variable that the
                # ensemble endpoint only serves deterministically).
                col = daily.get(var)
                v = col[i] if (col is not None and i < len(col)) else None
                if v is None:
                    continue
                stat = {"mean": float(v), "p10": None, "p90": None, "n": 0}

            if var in FULL_MEMBER_VARS and vals:
                stat["members"] = [round(x, 2) for x in vals]
            row[var] = stat
        if row:
            out[dt] = row
    return out


def _hourly_daily_wind(hourly):
    """{date: {u,v,speed,n_hours}} — vector-averaged daily mean, km/h.

    Averaging u and v (not speed and direction) is what keeps opposing
    directions from cancelling into a bogus 10 km/h "mean" vector, and it makes
    the daily wind a real mean rather than v19.4's windspeed_max x 0.50.
    """
    times = hourly.get("time") or []
    spd = hourly.get("wind_speed_10m") or []
    wdir = hourly.get("wind_direction_10m") or []
    acc = {}
    for i, ts in enumerate(times):
        if i >= len(spd) or i >= len(wdir):
            break
        s, d = spd[i], wdir[i]
        if s is None or d is None:
            continue
        u = -float(s) * math.sin(math.radians(float(d)))
        v = -float(s) * math.cos(math.radians(float(d)))
        a = acc.setdefault(ts.split("T")[0], [0.0, 0.0, 0.0, 0])
        a[0] += u
        a[1] += v
        a[2] += float(s)
        a[3] += 1
    return {
        dt: {"u": a[0] / a[3], "v": a[1] / a[3], "speed": a[2] / a[3], "n_hours": a[3]}
        for dt, a in acc.items() if a[3]
    }


def _needs_supplement(bundle, cities, model_key):
    """Variables this model has no value for anywhere (e.g. ICON CAPE/gusts)."""
    seen = set()
    for city in cities:
        for day in bundle[city].values():
            entry = day.get(model_key)
            if entry:
                seen.update(entry["vars"])
    return [v for v in DAILY_VARS if v not in seen]


def fetch_all_models(locs):
    """Fetch every model once for the whole domain. Returns (data, meta)."""
    cities = list(locs)
    coords = list(locs.values())
    target_dates = [d.strftime("%Y-%m-%d") for d in forecast_days]
    bundle = {c: {} for c in cities}
    member_counts, det_only, wind_models = {}, set(), set()
    hourly_ok = []

    staging = {}
    if os.path.exists(STAGING_CACHE_FILE):
        try:
            with open(STAGING_CACHE_FILE, "r", encoding="utf-8") as sf:
                staging = json.load(sf)
        except Exception as e:
            log.debug(f"Staging cache unreadable: {e}")

    for mk, name in ENSEMBLE_MODELS.items():
        is_ens = mk in ENSEMBLE_ONLY_MODELS
        if mk in staging and (staging[mk].get("member_count", 0) > 0 or not is_ens):
            s_data = staging[mk].get("data", {})
            for c, days in s_data.items():
                if c in bundle:
                    for dt, m_entry in days.items():
                        bundle[c].setdefault(dt, {})[mk] = m_entry
            n_mem = staging[mk]["member_count"]
            member_counts[mk] = n_mem
            if n_mem == 0:
                det_only.add(mk)
            if staging[mk].get("wind"):
                wind_models.add(mk)
                hourly_ok.append(name)
            log.info(f"  📦 {name}: loaded from today's staging cache ({f'{n_mem} members' if n_mem else 'deterministic'})")
            continue

        target_url = ENSEMBLE_URL if is_ens else FORECAST_URL
        log.info(f"  ◇ {name}: {'ensemble' if is_ens else 'forecast'} daily, {len(cities)} locations, 1 request")
        evars = MODEL_ENSEMBLE_VARS.get(mk, DAILY_VARS)
        resp = _as_locations(
            _post_json(target_url, _multi_payload(
                coords, days=len(target_dates), daily=",".join(evars), models=mk), f"{name} daily"),
            len(cities), f"{name} daily")

        n_mem = 0
        if resp is not None:
            for city, loc in zip(cities, resp):
                if not isinstance(loc, dict) or "daily" not in loc:
                    continue
                for dt, row in _daily_stats(loc["daily"]).items():
                    if dt in target_dates:
                        bundle[city].setdefault(dt, {})[mk] = {"vars": row}
                        n_mem = max(n_mem, max((s.get("n", 0) for s in row.values()), default=0))

        # ── deterministic supplement for variables the ensemble path omits ──
        # AIFS/WN2 have no deterministic run at all (every field returns null),
        # so asking the forecast endpoint for them only burns a request; and if
        # their ensemble call failed there is nothing left to fall back to.
        if resp is None:
            if is_ens:
                log.error(f"    ✘ {name}: ensemble endpoint unavailable — model dropped")
                note_once(f"{name} dropped (ensemble endpoint unavailable)")
                continue
            missing = list(DAILY_VARS)
        elif is_ens:
            missing = []
        else:
            missing = _needs_supplement(bundle, cities, mk)
        if missing:
            log.info(f"    ↳ {name}: supplementing {', '.join(missing)} from forecast endpoint")
            sup = _as_locations(
                _post_json(FORECAST_URL, _multi_payload(
                    coords, days=len(target_dates), daily=",".join(missing), models=mk), f"{name} supplement"),
                len(cities), f"{name} supplement")
            if sup is not None:
                for city, loc in zip(cities, sup):
                    if not isinstance(loc, dict) or "daily" not in loc:
                        continue
                    for dt, row in _daily_stats(loc["daily"]).items():
                        if dt in target_dates:
                            entry = bundle[city].setdefault(dt, {}).setdefault(mk, {"vars": {}})
                            for var, st in row.items():
                                if st.get("n", 0) == 0:   # keep member stats if any
                                    entry["vars"].setdefault(var, st)

        member_counts[mk] = n_mem
        if n_mem == 0:
            det_only.add(mk)
            if is_ens:
                log.warning(f"    ⚠ {name}: no ensemble members returned")
            else:
                log.info(f"    ✓ {name}: deterministic (forecast endpoint)")
        else:
            log.info(f"    ✓ {name}: {n_mem} members")

        # ── hourly wind → daily-mean vector ──────────────────────────────────
        if mk not in WIND_MODELS:
            continue
        wlabel = f"{name} hourly wind"
        wresp = _as_locations(
            _post_json(FORECAST_URL, _multi_payload(
                coords, days=len(target_dates),
                hourly="wind_speed_10m,wind_direction_10m", models=mk), wlabel),
            len(cities), wlabel)
        got = False
        if wresp is not None:
            for city, loc in zip(cities, wresp):
                if not isinstance(loc, dict) or "hourly" not in loc:
                    continue
                for dt, w in _hourly_daily_wind(loc["hourly"]).items():
                    if dt not in target_dates:
                        continue
                    entry = bundle[city].setdefault(dt, {}).setdefault(mk, {"vars": {}})
                    entry["wind"] = w
                    got = True
        if got:
            wind_models.add(mk)
            hourly_ok.append(name)
        else:
            log.warning(f"    ⚠ {name}: no hourly wind — excluded from the vector wind field")

        if n_mem > 0 or not is_ens:
            mk_bundle = {
                c: {dt: bundle[c][dt][mk] for dt in bundle[c] if mk in bundle[c][dt]}
                for c in cities
            }
            staging[mk] = {
                "data": mk_bundle,
                "member_count": n_mem,
                "wind": (mk in wind_models),
            }
            try:
                with open(STAGING_CACHE_FILE, "w", encoding="utf-8") as sf:
                    json.dump(_json_safe(staging), sf)
            except Exception as se:
                log.debug(f"Failed writing staging cache: {se}")

    # Drop model/day combinations that produced nothing at all.
    for city in cities:
        for dt in list(bundle[city]):
            if not bundle[city][dt]:
                del bundle[city][dt]

    meta = {
        "version": CACHE_SCHEMA_VERSION,
        "created": RUN_TIMESTAMP,
        "member_counts": member_counts,
        "det_only": sorted(det_only),
        "wind_models": sorted(wind_models),
        "models": list(ENSEMBLE_MODELS),
        "coverage": _model_coverage(bundle, cities),
    }
    return bundle, meta


def _model_coverage(bundle, cities):
    """{model: [dates present at >=90% of cities]} — the model's real horizon.

    Needed because ICON's ensemble endpoint stops at +7 d while the others run
    to +10 d; a naive "all models on all days" cache check would therefore
    never pass and would refetch on every run.
    """
    counts = {}
    for city in cities:
        for dt, models in bundle[city].items():
            for mk in models:
                counts.setdefault(mk, Counter())[dt] += 1
    need = 0.9 * len(cities)
    return {mk: sorted(dt for dt, n in c.items() if n >= need) for mk, c in counts.items()}


def _json_safe(obj):
    """NaN/Inf -> None so the cache is valid JSON for any consumer (not just Python)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.floating, np.integer)):
        return _json_safe(obj.item())
    return obj


def _cache_is_usable(cached, locs, meta):
    if not isinstance(cached, dict) or meta.get("version") != CACHE_SCHEMA_VERSION:
        return False
    data = cached.get("data") or {}
    if set(data) != set(locs):
        return False
    coverage = meta.get("coverage") or {}
    if not coverage:
        return False
    if set(coverage) != set(meta.get("models") or ENSEMBLE_MODELS):
        log.info(f"  Cache incomplete: missing model(s) "
                 f"{sorted(set(meta.get('models') or ENSEMBLE_MODELS) - set(coverage))}")
        return False
    # An ensemble-only model that produced no members is a degraded run:
    # the ensemble statistics would be partial.
    if any(m in meta.get("det_only", []) for m in ENSEMBLE_ONLY_MODELS):
        return False
    # Every city must carry every model on every day that model actually covers,
    # and the whole domain must carry every model on at least one day.
    for mk, dates in coverage.items():
        for city, days in data.items():
            missing = [d for d in dates if mk not in (days.get(d) or {})]
            if missing:
                log.info(f"  Cache incomplete: {city}/{mk} missing {missing[:3]}")
                return False
    for city, days in data.items():
        if not days:
            return False
        for mk in coverage:
            if not any(mk in (days.get(d) or {}) for d in days):
                log.info(f"  Cache incomplete: {city} has no {mk} at all")
                return False
    return True


def load_or_fetch(locs):
    """Return (data, meta). Only complete, schema-valid caches are reused."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cached = json.load(f)
            meta = cached.get("_meta", {})
            if _cache_is_usable(cached, locs, meta):
                log.info("  📦 Loading complete API data from local cache")
                return cached["data"], meta
            log.info("  ♻ Cache rejected (incomplete or old schema) — refetching")
        except Exception as exc:
            log.warning(f"  Cache unreadable ({exc}) — refetching")

    log.info("  🌐 Fetching all models in batched multi-location requests...")
    data, meta = fetch_all_models(locs)

    missing = [ENSEMBLE_MODELS[m] for m in ENSEMBLE_ONLY_MODELS if m in meta["det_only"]]
    if missing:
        note_once("degraded: " + ", ".join(missing) + " = deterministic run only")
    # Cache only a run in which every model covers every declared day at every city.
    complete = _cache_is_usable({"_meta": meta, "data": data}, locs, meta)
    if complete:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_json_safe({"_meta": meta, "data": data}), f)
        log.info(f"  💾 Cache written → {CACHE_FILE}")
        if os.path.exists(STAGING_CACHE_FILE):
            try:
                os.remove(STAGING_CACHE_FILE)
            except Exception:
                pass
    else:
        # A degraded run must never be replayed as complete tomorrow.
        note_once("incomplete run — not cached; ensemble statistics may be partial")
    return data, meta

# ── 5. STATS & SCORING ────────────────────────────────────────────────────────
def _stat_ok(stat):
    return isinstance(stat, dict) and stat.get("mean") is not None


def _resolve_weights(var, models):
    """Renormalise the configured weights over the models actually available."""
    wmap = VAR_WEIGHTS.get(var, MODEL_WEIGHTS)
    ws = {m: float(wmap.get(m, MODEL_WEIGHTS.get(m, 0.20))) for m in models}
    total = sum(ws.values())
    if total <= 0:
        return {m: 1.0 / len(models) for m in models}
    return {m: w / total for m, w in ws.items()}


def _weighted_percentile(values, weights, q):
    """Pooled weighted quantile (the standard multi-model ensemble definition)."""
    vals = np.asarray(values, dtype=float)
    wts = np.asarray(weights, dtype=float)
    ok = np.isfinite(vals) & np.isfinite(wts) & (wts > 0)
    vals, wts = vals[ok], wts[ok]
    if vals.size == 0:
        return np.nan
    order = np.argsort(vals)
    vals, wts = vals[order], wts[order]
    cw = np.cumsum(wts)
    cw /= cw[-1]
    return float(np.interp(q / 100.0, cw, vals))


def _homogeneous_blend(getter, cities, min_frac=0.95):
    """Fix ONE model set for a variable across the whole domain for this day.

    v19.4 renormalised weights per station, so stations fed by 3 models and
    stations fed by 5 models entered the same RBF with different estimators —
    a genuine discontinuity in the interpolant's input, not a cosmetic issue.
    """
    present = {c: {m for m in ENSEMBLE_MODELS if getter(c, m) is not None} for c in cities}
    counts = Counter()
    for c in cities:
        counts.update(present[c])
    eligible = {m for m, k in counts.items() if k >= min_frac * len(cities)}
    if not eligible:
        return set(), []
    # Never thin a variable silently: say which model failed homogeneity.
    for m in ENSEMBLE_MODELS:
        if 0 < counts.get(m, 0) < min_frac * len(cities):
            log.debug(f"    {ENSEMBLE_MODELS[m]} excluded from a variable: "
                      f"only {counts[m]}/{len(cities)} stations")
    cities_ok = [c for c in cities if eligible <= present[c]]
    # RBFInterpolator needs >=4 points; on a tiny test domain that is the whole set.
    return eligible, (cities_ok if len(cities_ok) >= min(4, len(cities)) else [])


def compute_day_stats(day_data):
    """Blended statistics for every city, from a homogeneous model set per variable.

    ``day_data``: {city: {model: {"vars": {var: {mean,p10,p90,n,members}}, "wind": {...}}}
    """
    cities = [c for c in day_data if day_data[c]]
    stats = {c: {} for c in cities}
    used_models = set()
    ctx = {}   # var -> (model set, cities admitted) — reused for derived variables

    def value_getter(var):
        def get(city, model):
            entry = day_data[city].get(model) or {}
            if var in ("u_wind", "v_wind"):
                w = entry.get("wind")
                return None if not w else (w["u"] if var == "u_wind" else w["v"])
            st = entry.get("vars", {}).get(var)
            return st["mean"] if _stat_ok(st) else None
        return get

    for var in DAILY_VARS + ["u_wind", "v_wind"]:
        if var == "winddirection_10m_dominant":
            continue   # derived from the blended vector, never averaged as an angle

        getter = value_getter(var)
        models, cities_ok = _homogeneous_blend(getter, cities)
        if not models or not cities_ok:
            for c in cities:
                stats[c][var] = np.nan
                stats[c][var + "_spread"] = np.nan
                stats[c][var + "_nmodels"] = 0
            continue

        used_models |= models
        ctx[var] = (models, set(cities_ok))
        w = _resolve_weights(var, models)
        for c in cities:
            # Cities outside the homogeneous subset are excluded from this
            # variable entirely rather than entering the RBF with fewer models.
            if c not in cities_ok:
                stats[c][var] = np.nan
                stats[c][var + "_spread"] = np.nan
                stats[c][var + "_nmodels"] = 0
                continue
            available = {m: getter(c, m) for m in models if getter(c, m) is not None}
            stats[c][var + "_nmodels"] = len(available)
            if not available:
                stats[c][var] = np.nan
                stats[c][var + "_spread"] = np.nan
                continue
            ww = {m: w[m] for m in available}
            tot = sum(ww.values())
            stats[c][var] = float(sum(available[m] * ww[m] for m in available) / tot)

            # Ensemble spread = member P90-P10, weighted across models.
            # (v19.4 stored max-min across models, i.e. model disagreement,
            #  and never used it. This is a real ensemble spread.)
            spreads, sw = [], []
            for m in available:
                st = day_data[c][m]["vars"].get(var) or {}
                if st.get("n", 0) > 0 and st.get("p90") is not None:
                    spreads.append(st["p90"] - st["p10"])
                    sw.append(ww[m])
            stats[c][var + "_spread"] = (
                float(np.average(spreads, weights=sw)) if spreads else np.nan)

    # ── wind vector + direction from the blended components ──────────────────
    if not any(np.isfinite(stats[c].get("u_wind") or np.nan) for c in cities):
        note_once("daily convergence unavailable (no hourly wind vectors)")
    for c in cities:
        u, v = stats[c].get("u_wind"), stats[c].get("v_wind")
        if u is not None and v is not None and np.isfinite(u) and np.isfinite(v):
            stats[c]["winddirection_10m_dominant"] = (
                math.degrees(math.atan2(-u, -v)) + 360) % 360
        else:
            stats[c]["u_wind"] = stats[c]["v_wind"] = np.nan
            stats[c]["winddirection_10m_dominant"] = np.nan

    # ── precipitation: pooled member P90 / probability / max / spread ────────
    p_models, p_cities = _homogeneous_blend(value_getter("precipitation_sum"), cities)
    p_w = _resolve_weights("precipitation_sum", p_models) if p_models else {}
    precip_stats = {}
    for c in p_cities:
        vals, wts = [], []
        for m in p_models:
            st = day_data[c][m]["vars"].get("precipitation_sum") or {}
            mem = st.get("members") or []
            if not mem:
                continue
            wts.extend([p_w[m] / len(mem)] * len(mem))
            vals.extend(mem)
        if not vals:
            precip_stats[c] = (np.nan, np.nan, np.nan, np.nan)
            continue
        arr = np.asarray(vals, dtype=float)
        wts = np.asarray(wts, dtype=float)
        prob = 100.0 * wts[arr > PRECIP_PROB_THRESHOLD_MM].sum() / wts.sum()
        precip_stats[c] = (
            _weighted_percentile(arr, wts, 90),
            float(arr.max()),
            float(prob),
            _weighted_percentile(arr, wts, 90) - _weighted_percentile(arr, wts, 10),
        )
    for c in cities:
        p90, pmax, prob, pspr = precip_stats.get(c, (np.nan,) * 4)
        stats[c]["precip_p90"] = p90
        stats[c]["precip_max"] = pmax
        stats[c]["precip_prob"] = prob
        stats[c]["precip_spread"] = pspr

    # ── derived quantities, computed per model then blended ──────────────────
    # v19.4 applied Rothfusz to a 5-model mean Tmax and a 4-model mean dewpoint
    # from a different model population; compose-then-average is the correct one.
    # Both inputs are restricted to the intersection of the two homogeneous
    # model sets so the interpolated heat-index field is estimator-consistent too.
    tmax_ctx = ctx.get("temperature_2m_max")
    dp_ctx = ctx.get("dewpoint_2m_mean")
    for c in cities:
        stats[c]["apparent_temp"] = np.nan
        if not (tmax_ctx and dp_ctx):
            continue
        models = tmax_ctx[0] & dp_ctx[0]
        if c not in tmax_ctx[1] or c not in dp_ctx[1] or not models:
            continue
        hi_vals, hi_w = [], []
        for m in models:
            entry = day_data[c].get(m) or {}
            tmax_st = entry.get("vars", {}).get("temperature_2m_max")
            dp_st = entry.get("vars", {}).get("dewpoint_2m_mean")
            if not (_stat_ok(tmax_st) and _stat_ok(dp_st)):
                continue
            hi = compute_heat_index(tmax_st["mean"], dp_st["mean"])
            if np.isfinite(hi):
                hi_vals.append(hi)
                hi_w.append(MODEL_WEIGHTS.get(m, 0.2))
        if hi_vals:
            stats[c]["apparent_temp"] = float(np.average(hi_vals, weights=hi_w))

    # DTR likewise: per-model Tmax-Tmin (same model, same day), then blended.
    tmin_ctx = ctx.get("temperature_2m_min")
    for c in cities:
        stats[c]["dtr"] = np.nan
        if not (tmax_ctx and tmin_ctx):
            continue
        models = tmax_ctx[0] & tmin_ctx[0]
        if c not in tmax_ctx[1] or c not in tmin_ctx[1] or not models:
            continue
        vals, ws = [], []
        for m in models:
            entry = day_data[c].get(m) or {}
            hi_st = entry.get("vars", {}).get("temperature_2m_max")
            lo_st = entry.get("vars", {}).get("temperature_2m_min")
            if _stat_ok(hi_st) and _stat_ok(lo_st):
                vals.append(hi_st["mean"] - lo_st["mean"])
                ws.append(MODEL_WEIGHTS.get(m, 0.2))
        if vals:
            stats[c]["dtr"] = float(np.average(vals, weights=ws))

    return stats, used_models


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

def _station_distance_km():
    """Distance (km) from every grid cell to the nearest observing station."""
    station_cells = np.zeros(glat.shape, dtype=bool)
    for lat_val, lon_val in coords:
        iy = int(np.clip(np.searchsorted(gy, lat_val), 0, len(gy) - 1))
        ix = int(np.clip(np.searchsorted(gx, lon_val), 0, len(gx) - 1))
        station_cells[iy, ix] = True
    return ndimage.distance_transform_edt(~station_cells) * _grid_km_per_px()


# Confidence radius for a station-derived field. Beyond this the thin-plate
# spline extrapolates unbounded (v19.4 reached +33 km/h from a +-7 km/h
# station range), manufacturing coastal convergence bands out of nothing.
CONV_FULL_KM = 150.0
CONV_ZERO_KM = 220.0
_STATION_DIST_KM = None


def _convergence_taper():
    global _STATION_DIST_KM
    if _STATION_DIST_KM is None:
        _STATION_DIST_KM = _station_distance_km()
    return np.clip((CONV_ZERO_KM - _STATION_DIST_KM) / (CONV_ZERO_KM - CONV_FULL_KM), 0.0, 1.0)


def _convergence_vmax(field, mask_bool):
    """Colour half-range that actually uses the field's dynamic range.

    v19.4 fixed +-5 x10^-5 s^-1 against a measured p99 of ~2.7, so the daily
    product rendered as a near-uniform pale wash.
    """
    vals = np.abs(field[mask_bool & np.isfinite(field)])
    if vals.size == 0:
        return 1.0
    return float(np.clip(np.ceil(np.percentile(vals, 95) * 10) / 10, 0.3, 5.0))


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
    ] + (["⚠ " + " · ".join(RUN_NOTES)] if RUN_NOTES else [])
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

def _grid_km_per_px():
    """Mean cell size in km for this domain (equirectangular, mid-latitude).

    Grid is anisotropic here (~2.07 km E-W vs ~1.34 km N-S), so distance-based
    helpers use the mean; that is the honest resolution limit of station data
    anyway.
    """
    lat_mid = 0.5 * (miny + maxy)
    return 0.5 * (dlat_deg * 111.139 + dlon_deg * 111.139 * math.cos(math.radians(lat_mid)))


def _lpa_local_minima(field, radius_km=150.0):
    """Local minima on a NaN-masked MSLP field, NaN-safe.

    NaN terrain mask is replaced by a large value *before* filtering: otherwise
    minimum_filter propagates NaN through the whole window and no LPA within
    ~radius of the Western Ghats can ever be detected (v19.4 behaviour).
    """
    px = max(3, int(round(2 * radius_km / _grid_km_per_px())))
    filled = np.where(np.isnan(field), 1e6, field)
    return ndimage.minimum_filter(filled, size=px) == filled


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

    # Synoptic window defined in kilometres and converted with the real grid
    # geometry (this domain spans ~11.4 deg x ~7.2 deg on 600 px, i.e. ~2.07 km
    # per px E-W and ~1.34 km N-S, mean ~1.69 km). v19.4 hard-coded size=150 and
    # claimed "~180 km"; minimum_filter's `size` is the full window, so 150 was
    # a 75 px radius, i.e. ~100-155 km. The window is now derived, not guessed.
    local_min = _lpa_local_minima(masked_mslp)
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
    # Levels calibrated to the *reachable* distribution of a 0.25 deg
    # instantaneous IFS field over this domain: >1.5 : ~52% of cells,
    # >3.0 : ~17%, >5.0 : ~5%, >8.0 : 0%.  v19.4's 8.0 and 12.0 levels -- and
    # therefore both "strong convergence" legend entries -- were unreachable.
    levels = [1.5, 3.0, 5.0]
    linewidths = [1.5, 2.2, 3.0]
    colors = ["#74c476", "#31a354", "#54278f"]  # greens → deep purple

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

    # Custom Legend for the contour lines (thresholds must match `levels`)
    legend_elements = [
        plt.matplotlib.lines.Line2D((0,), (0,), color="#31a354", lw=2.2, label='Mod. Conv (>3.0 × 10⁻⁵ s⁻¹)'),
        plt.matplotlib.lines.Line2D((0,), (0,), color="#54278f", lw=3.0, label='Str. Conv (>5.0 × 10⁻⁵ s⁻¹)')
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
    """Install cfgrib only when explicitly opted in.

    v19.4 shelled out to `pip install --break-system-packages` as a side effect
    of a forecast run, which on the Conda/Windows layout this file already
    patches findlibs for is exactly the wrong thing to do.
    """
    if os.environ.get("MUGIL_ALLOW_PIP_INSTALL") != "1":
        log.error("  cfgrib/eccodes unavailable — set MUGIL_ALLOW_PIP_INSTALL=1 to auto-install")
        return False
    import subprocess, sys
    log.info("  cfgrib not found — attempting auto-install (MUGIL_ALLOW_PIP_INSTALL=1)...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "cfgrib", "-q"],
            capture_output=True, timeout=120
        )
        if result.returncode == 0:
            import cfgrib  # noqa: F401
            log.info("  ✓ cfgrib installed successfully.")
            return True
        log.warning(f"  pip install cfgrib failed:\n{result.stderr.decode().strip()}")
    except Exception as exc:
        log.warning(f"  Auto-install failed: {exc}")
    return False

def process_grib_file(grib_path, var_name="2t", apply_mask=False):
    """Read one field from the GRIB and interpolate it to the render grid.

    ``apply_mask`` defaults to False: callers must mask *after* differentiating.
    v19.4 masked to land before np.gradient, which turned the entire coastal
    band (the meteorologically interesting part) into NaN and then bled it
    ~2 px inland through the gaussian smoother.
    """
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

    except Exception as exc:
        log.error(f"  Failed to read GRIB file: {exc}")
        return None

# ── 10. MAIN EXECUTION LOOP ───────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("📡 Step 1: Fetching all models in batched multi-location requests.")

    elevs      = get_elevations(coords)
    city_elev_km = {name: elevs[i] / 1000.0 for i, name in enumerate(locations)}
    daily_rows = {dt.strftime("%Y-%m-%d"): [] for dt in forecast_days}

    # ── fetch (5-15 HTTP requests for the whole domain) ──────────────────────
    all_city_data, cache_meta = load_or_fetch(locations)

    # ── group into per-day {city: {model: {...}}} so one model set can be
    #    fixed per (day, variable) across the whole domain ───────────────────
    for day_idx, dt in enumerate(forecast_days):
        dt_str = dt.strftime("%Y-%m-%d")
        day_data = {}
        for name in locations:
            entry = (all_city_data.get(name) or {}).get(dt_str)
            if entry:
                day_data[name] = entry
        if not day_data:
            log.warning(f"  No model data for {dt_str}")
            continue

        try:
            stats_by_city, used = compute_day_stats(day_data)
        except Exception as exc:
            log.error(f"  Stats error for {dt_str}: {exc}")
            continue

        for name, (lat, lon) in locations.items():
            stats = stats_by_city.get(name)
            if not stats:
                continue
            try:
                tmax_val = stats.get("temperature_2m_max", np.nan)
                tmin_val = stats.get("temperature_2m_min", np.nan)
                dp_val   = stats.get("dewpoint_2m_mean", np.nan)

                daily_rows[dt_str].append({
                    "City":     name,
                    "Lat":      lat,
                    "Lon":      lon,
                    "StationElev": city_elev_km[name],
                    "MaxTemp":  tmax_val,
                    "MinTemp":  tmin_val,
                    "DTR":      stats.get("dtr", np.nan),
                    "Precip":   stats.get("precipitation_sum", np.nan),
                    "Precip_P90": stats.get("precip_p90", np.nan),
                    "Precip_Max": stats.get("precip_max", np.nan),
                    "Precip_Spread": stats.get("precip_spread", np.nan),
                    "Gusts":    stats.get("windgusts_10m_max", np.nan),
                    "Gusts_Spread": stats.get("windgusts_10m_max_spread", np.nan),
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
                    "ApparentTemp": stats.get("apparent_temp", np.nan),
                })
            except Exception as exc:
                log.error(f"  Row error for {name} on {dt_str}: {exc}")

        # header must describe the models actually blended, not a fixed table
        set_model_label(used, cache_meta.get("member_counts", {}),
                        set(cache_meta.get("det_only", ())))

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
        save_map(grid_precip, "viridis", "Precipitation P90 — Ensemble (mm/day)", f"02_precip_{tag}.png",
                 df, "Precip_P90", "mm/day", date_label, is_precip=True)

        # True member-based probability of rain (v19.4 computed a 5-sample
        # model vote and never mapped it). White when no member exceeds 1 mm.
        z_prob = smooth_interpolation(lons_d, lats_d, df.PrecipProb.values, kernel='linear')
        grid_prob = np.where(mask, z_prob, np.nan)
        save_map(grid_prob, "Blues", "Probability of Rain > 1 mm (ensemble, %)", f"02b_precip_prob_{tag}.png",
                 df, "PrecipProb", "%", date_label, fixed_vmin=0, fixed_vmax=100)

        # Ensemble spread (P90-P10 of the pooled members), not model disagreement.
        z_pspr = smooth_interpolation(lons_d, lats_d, df.Precip_Spread.values, kernel='linear')
        grid_pspr = np.where(mask, z_pspr, np.nan)
        save_map(grid_pspr, "YlGnBu", "Precipitation Ensemble Spread P90−P10 (mm/day)",
                 f"02c_precip_spread_{tag}.png", df, "Precip_Spread", "mm/day", date_label)

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

            # Taper out the TPS tail: beyond the station confidence radius the
            # spline extrapolates unbounded and invents offshore convergence.
            grid_conv = np.where(mask, smoothed_conv * _convergence_taper(), np.nan)

            df["Convergence"] = np.nan
            conv_vmax = _convergence_vmax(grid_conv, mask)
            save_map(grid_conv, "PiYG",
                     f"Wind Convergence (×10⁻⁵ s⁻¹)  [+ve = Lift]  ·  scale ±{conv_vmax:g}",
                     f"09_convergence_{tag}.png",
                     df, "Convergence", "", date_label,
                     fixed_vmin=-conv_vmax, fixed_vmax=conv_vmax, is_centered=True)
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
        grid_u10 = process_grib_file(temp_grib_path, "u10", apply_mask=False)
        grid_v10 = process_grib_file(temp_grib_path, "v10", apply_mask=False)

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
        # cfgrib writes .idx sidecars; v19.4 left them littering the repo
        for sidecar in (temp_grib_path + ".idx", temp_grib_path + ".5b7b6.idx"):
            if os.path.exists(sidecar):
                os.remove(sidecar)

    log.info("\n🌊 Step 4: Generating Oceanic Diagnostics (IOD & Pacific SST for El Niño/La Niña)...")
    try:
        from mugil_sst_maps import generate_ocean_sst_maps
        generate_ocean_sst_maps(OUT_DIR, RUN_TIMESTAMP)
    except Exception as e:
        log.warning(f"  ✘ Ocean SST map generation failed or bypassed: {e}")

    log.info(f"\n✅ Pipeline complete → '{OUT_DIR}/'")
