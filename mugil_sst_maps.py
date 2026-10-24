"""
MUGIL Ensemble Ocean Surface Temperature Module
================================================
Generates high-resolution Sea Surface Temperature (SST) maps:
1. Indian Ocean Dipole (IOD) Map:
   - Covers South India, Arabian Sea, and Bay of Bengal.
   - Land mass fully masked (no plots on land mass).
   - Shows ocean surface thermal structure and regional SST averages.
2. Pacific Ocean SST Map:
   - Full Pacific basin view for tracking El Niño & La Niña (ENSO).
   - Highlights equatorial Pacific with Niño 1+2, Niño 3, Niño 3.4, and Niño 4 monitoring boxes.
   - Displays broad latitudinal extent reaching sub-polar and polar boundary margins.
Data Source: NOAA Daily Optimum Interpolation Sea Surface Temperature (OISST v2.1).
"""

import os
import time
import requests
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from datetime import datetime, timezone

# Load locally cached Natural Earth shapefiles
try:
    land_shp = shpreader.natural_earth('50m', 'physical', 'land')
    land_geoms = list(shpreader.Reader(land_shp).geometries())
except Exception:
    land_geoms = []

try:
    borders_shp = shpreader.natural_earth('50m', 'cultural', 'admin_0_boundary_lines_land')
    borders_geoms = list(shpreader.Reader(borders_shp).geometries())
except Exception:
    borders_geoms = []

try:
    states_shp = shpreader.natural_earth('10m', 'cultural', 'admin_1_states_provinces_lines')
    states_geoms = list(shpreader.Reader(states_shp).geometries())
except Exception:
    states_geoms = []


def fetch_iod_sst():
    """Fetch OISST / CoralTemp for Arabian Sea & Bay of Bengal (0-25N, 55-98E)."""
    urls = [
        "https://coastwatch.noaa.gov/erddap/griddap/noaacrwsstDaily.json?analysed_sst[(last)][(0.0):5:(25.0)][(55.0):5:(98.0)]",
        "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21NrtAgg_LonPM180.json?sst[(last)][(0.0)][(0.0):2:(25.0)][(55.0):2:(98.0)]",
        "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180.json?sst[(last)][(0.0)][(0.0):2:(25.0)][(55.0):2:(98.0)]",
    ]
    r = None
    for url in urls:
        for attempt in range(2):
            try:
                r = requests.get(url, timeout=25)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(1)
        if r is not None and r.status_code == 200:
            break

    if r is None or r.status_code != 200:
        raise RuntimeError(f"Failed to fetch IOD SST from NOAA ERDDAP (status {r.status_code if r else 'None'})")

    data = r.json()
    cols = data['table']['columnNames']
    lat_idx_col = cols.index('latitude')
    lon_idx_col = cols.index('longitude')
    sst_idx_col = cols.index('analysed_sst') if 'analysed_sst' in cols else cols.index('sst')

    rows = data['table']['rows']
    raw_date = rows[0][0][:10]
    dt = datetime.strptime(raw_date, "%Y-%m-%d")
    date_str = dt.strftime("%d %b %Y")

    lats = np.array(sorted(list(set(row[lat_idx_col] for row in rows))))
    lons = np.array(sorted(list(set(row[lon_idx_col] for row in rows))))

    lat_idx = {lat: i for i, lat in enumerate(lats)}
    lon_idx = {lon: j for j, lon in enumerate(lons)}

    sst_grid = np.full((len(lats), len(lons)), np.nan, dtype=float)
    for row in rows:
        lat = row[lat_idx_col]
        lon = row[lon_idx_col]
        val = row[sst_idx_col]
        if val is not None:
            sst_grid[lat_idx[lat], lon_idx[lon]] = val

    return lons, lats, sst_grid, date_str


def fetch_pacific_sst():
    """Fetch OISST / CoralTemp for Pacific Ocean (-60 to 65N, 120 to 280E)."""
    # 1. Primary: coastwatch.noaa.gov (CoralTemp 5km global, fast and reliable)
    try:
        url_west = "https://coastwatch.noaa.gov/erddap/griddap/noaacrwsstDaily.json?analysed_sst[(last)][(-60.0):10:(65.0)][(120.0):10:(180.0)]"
        url_east = "https://coastwatch.noaa.gov/erddap/griddap/noaacrwsstDaily.json?analysed_sst[(last)][(-60.0):10:(65.0)][(-180.0):10:(-80.0)]"
        rw = requests.get(url_west, timeout=25)
        re = requests.get(url_east, timeout=25)
        if rw.status_code == 200 and re.status_code == 200:
            data_w = rw.json()
            data_e = re.json()
            cols = data_w['table']['columnNames']
            lat_c = cols.index('latitude')
            lon_c = cols.index('longitude')
            sst_c = cols.index('analysed_sst') if 'analysed_sst' in cols else cols.index('sst')
            rows_w = data_w['table']['rows']
            rows_e = data_e['table']['rows']
            for row in rows_e:
                row[lon_c] += 360.0
            all_rows = rows_w + rows_e
            raw_date = all_rows[0][0][:10]
            dt = datetime.strptime(raw_date, "%Y-%m-%d")
            date_str = dt.strftime("%d %b %Y")
            lats = np.array(sorted(list(set(row[lat_c] for row in all_rows))))
            lons = np.array(sorted(list(set(row[lon_c] for row in all_rows))))
            lat_idx = {lat: i for i, lat in enumerate(lats)}
            lon_idx = {lon: j for j, lon in enumerate(lons)}
            sst_grid = np.full((len(lats), len(lons)), np.nan, dtype=float)
            for row in all_rows:
                val = row[sst_c]
                if val is not None:
                    sst_grid[lat_idx[row[lat_c]], lon_idx[row[lon_c]]] = val
            return lons, lats, sst_grid, date_str
    except Exception:
        pass

    # 2. Fallback: coastwatch.pfeg.noaa.gov (0-360 OISST)
    urls = [
        "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21NrtAgg.json?sst[(last)][(0.0)][(-60.0):6:(65.0)][(120.0):6:(280.0)]",
        "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg.json?sst[(last)][(0.0)][(-60.0):6:(65.0)][(120.0):6:(280.0)]",
    ]
    r = None
    for url in urls:
        for attempt in range(2):
            try:
                r = requests.get(url, timeout=35)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(1)
        if r is not None and r.status_code == 200:
            break

    if r is None or r.status_code != 200:
        raise RuntimeError(f"Failed to fetch Pacific SST from NOAA ERDDAP (status {r.status_code if r else 'None'})")

    data = r.json()
    cols = data['table']['columnNames']
    lat_idx_col = cols.index('latitude')
    lon_idx_col = cols.index('longitude')
    sst_idx_col = cols.index('sst') if 'sst' in cols else cols.index('analysed_sst')

    rows = data['table']['rows']
    raw_date = rows[0][0][:10]
    dt = datetime.strptime(raw_date, "%Y-%m-%d")
    date_str = dt.strftime("%d %b %Y")

    lats = np.array(sorted(list(set(row[lat_idx_col] for row in rows))))
    lons = np.array(sorted(list(set(row[lon_idx_col] for row in rows))))

    lat_idx = {lat: i for i, lat in enumerate(lats)}
    lon_idx = {lon: j for j, lon in enumerate(lons)}

    sst_grid = np.full((len(lats), len(lons)), np.nan, dtype=float)
    for row in rows:
        lat = row[lat_idx_col]
        lon = row[lon_idx_col]
        val = row[sst_idx_col]
        if val is not None:
            sst_grid[lat_idx[lat], lon_idx[lon]] = val

    return lons, lats, sst_grid, date_str


def render_iod_map(lons, lats, sst_grid, date_str, out_path, run_timestamp=None):
    """Render IOD Sea Surface Temperature map over Arabian Sea & Bay of Bengal."""
    if run_timestamp is None:
        run_timestamp = datetime.now().strftime("%d %b %Y, %H:%M IST")

    fig_w, fig_h = 10.5, 9.0
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#ffffff")

    map_left, map_bottom = 0.08, 0.12
    map_w, map_h = 0.80, 0.73
    ax = fig.add_axes([map_left, map_bottom, map_w, map_h], projection=ccrs.PlateCarree())

    # Bounds covering South India, Arabian Sea, and Bay of Bengal
    ax.set_extent([58, 96, 2, 24], crs=ccrs.PlateCarree())

    plot_sst = np.ma.masked_invalid(sst_grid)

    # Compute regional averages for context
    # Arabian Sea: 10N-20N, 60E-72E
    as_mask = (lats[:, None] >= 10) & (lats[:, None] <= 20) & (lons[None, :] >= 60) & (lons[None, :] <= 72)
    as_vals = plot_sst[as_mask]
    as_mean = float(np.nanmean(as_vals)) if len(as_vals.compressed()) > 0 else np.nan

    # Bay of Bengal: 10N-20N, 82E-92E
    bob_mask = (lats[:, None] >= 10) & (lats[:, None] <= 20) & (lons[None, :] >= 82) & (lons[None, :] <= 92)
    bob_vals = plot_sst[bob_mask]
    bob_mean = float(np.nanmean(bob_vals)) if len(bob_vals.compressed()) > 0 else np.nan

    vmin, vmax = 25.0, 31.5
    im = ax.pcolormesh(
        lons, lats, plot_sst,
        cmap='turbo', vmin=vmin, vmax=vmax,
        transform=ccrs.PlateCarree(),
        shading='auto', zorder=1
    )

    # White isotherms
    try:
        cs = ax.contour(
            lons, lats, plot_sst,
            levels=np.arange(26.0, 32.0, 1.0),
            colors='white', linewidths=0.9, alpha=0.8,
            transform=ccrs.PlateCarree(), zorder=2
        )
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.0f°C')
    except Exception:
        pass

    # Solid land mask: NO PLOTS ON LAND MASS
    if land_geoms:
        ax.add_geometries(land_geoms, ccrs.PlateCarree(), facecolor='#2d3748', edgecolor='#1a202c', linewidth=0.8, zorder=3)
    if borders_geoms:
        ax.add_geometries(borders_geoms, ccrs.PlateCarree(), facecolor='none', edgecolor='#4a5568', linewidth=0.6, zorder=4)
    if states_geoms:
        ax.add_geometries(states_geoms, ccrs.PlateCarree(), facecolor='none', edgecolor='#4a5568', linewidth=0.5, linestyle=':', zorder=4)

    # Coordinate gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='#cbd5e1', alpha=0.6, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 9, 'color': '#374151'}
    gl.ylabel_style = {'size': 9, 'color': '#374151'}

    # Oceanic Basin Annotations
    as_text = f"ARABIAN SEA\n({as_mean:.1f}°C avg)" if not np.isnan(as_mean) else "ARABIAN SEA"
    ax.text(66, 15, as_text, color="#ffffff", fontsize=12, weight="black",
            ha="center", va="center", transform=ccrs.PlateCarree(), zorder=5,
            path_effects=[pe.withStroke(linewidth=3, foreground="#111827")])

    bob_text = f"BAY OF BENGAL\n({bob_mean:.1f}°C avg)" if not np.isnan(bob_mean) else "BAY OF BENGAL"
    ax.text(88, 14, bob_text, color="#ffffff", fontsize=12, weight="black",
            ha="center", va="center", transform=ccrs.PlateCarree(), zorder=5,
            path_effects=[pe.withStroke(linewidth=3, foreground="#111827")])

    ax.text(77, 4.5, "EQUATORIAL INDIAN OCEAN", color="#ffffff", fontsize=11, weight="bold",
            ha="center", va="center", transform=ccrs.PlateCarree(), zorder=5,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="#111827")])

    # Colorbar
    cbar_ax = fig.add_axes([map_left + map_w + 0.02, map_bottom, 0.025, map_h])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label("Sea Surface Temperature (°C)", fontsize=10, fontweight="bold", labelpad=8)
    cb.ax.tick_params(labelsize=9)

    # Header and Footer
    _add_custom_header_footer(
        fig,
        title="INDIAN OCEAN DIPOLE (IOD) — SEA SURFACE TEMPERATURE",
        date_str=date_str,
        run_timestamp=run_timestamp,
        note=f"Arabian Sea Avg: {as_mean:.1f}°C  ·  Bay of Bengal Avg: {bob_mean:.1f}°C  ·  Land mass masked for marine thermal monitoring"
    )

    plt.savefig(out_path, dpi=200, bbox_inches=None, facecolor="#ffffff")
    plt.close(fig)


def render_pacific_map(lons, lats, sst_grid, date_str, out_path, run_timestamp=None):
    """Render Pacific Ocean SST map for tracking El Niño & La Niña."""
    if run_timestamp is None:
        run_timestamp = datetime.now().strftime("%d %b %Y, %H:%M IST")

    fig_w, fig_h = 12.0, 7.5
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#ffffff")

    map_left, map_bottom = 0.06, 0.12
    map_w, map_h = 0.83, 0.73
    ax = fig.add_axes([map_left, map_bottom, map_w, map_h], projection=ccrs.PlateCarree(central_longitude=180))

    # Extent centered on 180°: 120°E to 280°E (-60 to +100 in projection), -60 to 65 lat
    ax.set_extent([-60, 100, -60, 65], crs=ccrs.PlateCarree(central_longitude=180))

    plot_sst = np.ma.masked_invalid(sst_grid)

    # Compute Niño 3.4 SST average (5N-5S, 190E-240E / 170W-120W)
    n34_mask = (lats[:, None] >= -5) & (lats[:, None] <= 5) & (lons[None, :] >= 190) & (lons[None, :] <= 240)
    n34_vals = plot_sst[n34_mask]
    n34_mean = float(np.nanmean(n34_vals)) if len(n34_vals.compressed()) > 0 else np.nan

    vmin, vmax = -2.0, 31.0
    im = ax.pcolormesh(
        lons, lats, plot_sst,
        cmap='turbo', vmin=vmin, vmax=vmax,
        transform=ccrs.PlateCarree(),
        shading='auto', zorder=1
    )

    # Solid land mask
    if land_geoms:
        ax.add_geometries(land_geoms, ccrs.PlateCarree(), facecolor='#2d3748', edgecolor='#1a202c', linewidth=0.7, zorder=3)
    if borders_geoms:
        ax.add_geometries(borders_geoms, ccrs.PlateCarree(), facecolor='none', edgecolor='#4a5568', linewidth=0.5, zorder=4)

    # Equator line
    ax.plot([-60, 100], [0, 0], color='#ffffff', linestyle='--', linewidth=1.2, alpha=0.9,
            transform=ccrs.PlateCarree(central_longitude=180), zorder=4)
    ax.text(-58, 1.5, "Equator (0°)", color="#ffffff", fontsize=8.5, weight="bold",
            ha="left", va="bottom", transform=ccrs.PlateCarree(central_longitude=180), zorder=5,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="#111827")])

    # Draw ENSO Monitoring Boxes: Niño 4, Niño 3.4, Niño 3, Niño 1+2
    enso_boxes = [
        {"name": "Niño 4",   "lon": [160, 210], "lat": [-5, 5], "color": "#fbbf24"},
        {"name": "Niño 3.4", "lon": [190, 240], "lat": [-5, 5], "color": "#ef4444"},
        {"name": "Niño 3",   "lon": [210, 270], "lat": [-5, 5], "color": "#38bdf8"},
        {"name": "Niño 1+2", "lon": [270, 280], "lat": [-10, 0], "color": "#a855f7"}
    ]

    for b in enso_boxes:
        w = b["lon"][1] - b["lon"][0]
        h = b["lat"][1] - b["lat"][0]
        rect = mpatches.Rectangle(
            (b["lon"][0], b["lat"][0]), w, h,
            fill=False, edgecolor=b["color"], linewidth=2.0, linestyle='-',
            transform=ccrs.PlateCarree(), zorder=5
        )
        ax.add_patch(rect)
        label_y = b["lat"][1] + 2.0 if b["name"] != "Niño 3.4" else b["lat"][1] + 3.8
        ax.text(
            (b["lon"][0] + b["lon"][1]) / 2, label_y, b["name"],
            color=b["color"], fontsize=9.5, weight="black", ha="center", va="bottom",
            transform=ccrs.PlateCarree(), zorder=6,
            path_effects=[pe.withStroke(linewidth=3, foreground="#111827")]
        )

    # High-Latitude / Polar Boundary Regions
    ax.text(180, 58, "BERING SEA / SUB-ARCTIC PACIFIC", color="#ffffff", fontsize=10, weight="bold",
            ha="center", va="center", transform=ccrs.PlateCarree(), zorder=5,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="#111827")])

    ax.text(180, -54, "SOUTHERN OCEAN / SUB-ANTARCTIC MARGIN", color="#ffffff", fontsize=10, weight="bold",
            ha="center", va="center", transform=ccrs.PlateCarree(), zorder=5,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="#111827")])

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='#cbd5e1', alpha=0.4, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 8.5, 'color': '#374151'}
    gl.ylabel_style = {'size': 8.5, 'color': '#374151'}

    # Colorbar
    cbar_ax = fig.add_axes([map_left + map_w + 0.015, map_bottom, 0.02, map_h])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label("Sea Surface Temperature (°C)", fontsize=9.5, fontweight="bold", labelpad=8)
    cb.ax.tick_params(labelsize=8.5)

    n34_str = f"Niño 3.4 Mean SST: {n34_mean:.1f}°C" if not np.isnan(n34_mean) else "Niño 3.4 Region Active"
    _add_custom_header_footer(
        fig,
        title="PACIFIC OCEAN SST — EL NIÑO & LA NIÑA TRACKING",
        date_str=date_str,
        run_timestamp=run_timestamp,
        note=f"{n34_str}  ·  ENSO Index: > +0.5°C El Niño (Warm Phase) | < -0.5°C La Niña (Cool Phase)"
    )

    plt.savefig(out_path, dpi=200, bbox_inches=None, facecolor="#ffffff")
    plt.close(fig)


def _add_custom_header_footer(fig, title, date_str, run_timestamp, note):
    header_bot = 0.88
    footer_top = 0.08

    fig.add_artist(plt.matplotlib.lines.Line2D([0.0, 1.0], [header_bot, header_bot],
                   transform=fig.transFigure, color="#333333", linewidth=1.5, zorder=10))
    fig.add_artist(plt.matplotlib.lines.Line2D([0.0, 1.0], [footer_top, footer_top],
                   transform=fig.transFigure, color="#333333", linewidth=1.5, zorder=10))

    fig.text(0.04, 0.94, "MUGIL ENSEMBLE",
             ha="left", va="center", fontsize=18, color="#111111",
             fontweight="black", fontfamily="DejaVu Sans", zorder=10)

    fig.text(0.04, 0.90, f"{title}  •  {date_str}",
             ha="left", va="center", fontsize=10.5, color="#b22222",
             fontweight="bold", fontfamily="DejaVu Sans", zorder=10)

    fig.text(0.96, 0.94, "NOAA OISST v2.1 High-Resolution",
             ha="right", va="center", fontsize=9, color="#666666",
             fontfamily="DejaVu Sans", zorder=10)

    fig.text(0.50, 0.045, note,
             ha="center", va="center", fontsize=9, color="#000080",
             fontweight="bold", fontfamily="DejaVu Sans", zorder=10)

    fig.text(0.50, 0.02, f"⚠️ Experimental · Data: NOAA OISST · Generated: {run_timestamp}",
             ha="center", va="center", fontsize=7.5, color="#555555",
             fontstyle="italic", fontfamily="DejaVu Sans", zorder=10)


def generate_ocean_sst_maps(out_dir=".", run_timestamp=None):
    """Entry point to fetch and generate both IOD and Pacific SST maps."""
    os.makedirs(out_dir, exist_ok=True)
    iod_path = os.path.join(out_dir, "00_iod_arabian_sea_bob_sst.png")
    pacific_path = os.path.join(out_dir, "00_pacific_enso_sst.png")

    print("[1/2] Fetching & generating IOD Sea Surface Temperature map...")
    try:
        lons_iod, lats_iod, grid_iod, d_iod = fetch_iod_sst()
        render_iod_map(lons_iod, lats_iod, grid_iod, d_iod, iod_path, run_timestamp)
        print(f"  + Saved IOD map -> {iod_path}")
    except Exception as e:
        print(f"  - Failed to generate IOD map: {e}")

    print("[2/2] Fetching & generating Pacific El Nino / La Nina SST map...")
    try:
        lons_pac, lats_pac, grid_pac, d_pac = fetch_pacific_sst()
        render_pacific_map(lons_pac, lats_pac, grid_pac, d_pac, pacific_path, run_timestamp)
        print(f"  + Saved Pacific map -> {pacific_path}")
    except Exception as e:
        print(f"  - Failed to generate Pacific map: {e}")


if __name__ == '__main__':
    generate_ocean_sst_maps()
