import os
import time
import rasterio
import numpy as np
from datetime import date, timedelta
import re
import requests
from rasterio.warp import reproject, Resampling


# ✓ Iowa was the only one correct — all others recomputed from WGS84 → EPSG:5070
STATE_BBOXES = {
    "17": { "state": "ILLINOIS",      "bbox": "396000,1557000,732000,2203000"    },
    "18": { "state": "INDIANA",       "bbox": "689000,1666000,924000,2139000"    },
    "19": { "state": "IOWA",          "bbox": "-54000,1929000,471000,2293000"    },  
    "20": { "state": "KANSAS",        "bbox": "-533000,1567000,119000,1888000"   },
    "26": { "state": "MICHIGAN",      "bbox": "461000,2091000,1014000,2885000"   },
    "27": { "state": "MINNESOTA",     "bbox": "-100000,2279000,479000,2946000"   },
    "29": { "state": "MISSOURI",      "bbox": "21000,1438000,578000,1976000"     },
    "31": { "state": "NEBRASKA",      "bbox": "-680000,1916000,56000,2223000"    },
    "38": { "state": "NORTH DAKOTA",  "bbox": "-624000,2576000,-41000,2888000"   },
    "39": { "state": "OHIO",          "bbox": "965000,1764000,1268000,2212000"   },
    "46": { "state": "SOUTH DAKOTA",  "bbox": "-658000,2192000,-34000,2550000"   },
    "55": { "state": "WISCONSIN",     "bbox": "254000,2170000,743000,2715000"    },
}

# Official USDA CDL colour palette (subset)
CDL_LEGEND = {
    0:   ("Background",           "#000000"),1:   ("Corn",                 "#FFD300"),
    2:   ("Cotton",               "#FF2626"),3:   ("Rice",                 "#00A8E2"),
    4:   ("Sorghum",              "#FF9E0A"),5:   ("Soybeans",             "#267000"),
    6:   ("Sunflower",            "#FFD9D9"),21:  ("Barley",               "#A57000"),
    22:  ("Durum Wheat",          "#D3D300"),23:  ("Spring Wheat",         "#A5A500"),
    24:  ("Winter Wheat",         "#D3FF70"),26:  ("Dbl WinWht/Soy",       "#A5FF8C"),
    27:  ("Rye",                  "#70A500"),28:  ("Oats",                 "#70D000"),
    29:  ("Millet",               "#A57000"),36:  ("Alfalfa",              "#00AF4C"),
    37:  ("Other Hay",            "#00D000"),41:  ("Sugarbeets",           "#A800E2"),
    42:  ("Dry Beans",            "#A50000"),61:  ("Fallow/Idle",          "#B8AF93"),
    63:  ("Forest",               "#93CC93"),82:  ("Developed",            "#9C9C9C"),
    83:  ("Water",                "#4C70AF"),87:  ("Wetlands",             "#7FB3C8"),
    111: ("Open Water",           "#4C70AF"),121: ("Dev/Open Space",       "#9C9C9C"),
    122: ("Dev/Low",              "#9C9C9C"),123: ("Dev/Med",              "#9C9C9C"),
    124: ("Dev/High",             "#9C9C9C"),141: ("Deciduous Forest",     "#93CC93"),
    142: ("Evergreen Forest",     "#93CC93"),143: ("Mixed Forest",         "#93CC93"),
    152: ("Shrubland",            "#C6D69C"),176: ("Grassland/Pasture",    "#E8FFBE"),
    190: ("Woody Wetlands",       "#7FB3C8"),195: ("Herbaceous Wetlands",  "#7FB3C8"),
}


def download_cdl_tif(year, fips, out_path, retries=4, delay=20):
    """
    Download a state-level CDL GeoTIFF via the CropScape REST API.
    """
    if os.path.exists(out_path):
        print(f"[CDL {year}] Already exists: {out_path}")
        return True
        
    CROPSCAPE_BASE = "https://nassgeodata.gmu.edu/axis2/services/CDLService"
    request_url = f"{CROPSCAPE_BASE}/GetCDLFile?year={year}&fips={fips}"
    
    print(f"[CDL {year}] Requesting FIPS {fips} TIF from CropScape API...")
    print(f"  {request_url}")

    for attempt in range(1, retries + 1):
        try:
            r = requests.get(request_url, timeout=60)
            r.raise_for_status()

            match = re.search(r'<returnURL>(.*?)</returnURL>', r.text)
            if not match:
                print(f"  Attempt {attempt}: Unexpected response:\n  {r.text[:200]}")
                time.sleep(delay)
                continue

            tif_url = match.group(1).strip()
            print(f"  Got TIF URL: {tif_url}")
            print(f"  Downloading TIF (this may take a while for 10m res)...")
            
            tif_resp = requests.get(tif_url, timeout=300, stream=True)
            tif_resp.raise_for_status()

            with open(out_path, "wb") as f:
                for chunk in tif_resp.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)

            size_mb = os.path.getsize(out_path) / (1024 ** 2)
            print(f"  ✓ Saved {size_mb:.1f} MB → {out_path}")
            return True

        except Exception as e:
            print(f"  Attempt {attempt} error: {e}")
            time.sleep(delay * attempt)

    print(f"[CDL {year}] ✗ All {retries} attempts failed.")
    return False


def get_state_info(state_name, state_bboxes=STATE_BBOXES, state_fips=None):
    """
    Retrieve FIPS code and bounding box for a specified state name.
    """
    name_to_info = {
        v["state"].upper(): (k, v["bbox"])
        for k, v in state_bboxes.items()
    }

    key = state_name.upper().strip()

    if key in name_to_info:
        fips, bbox = name_to_info[key]
        print(f"FIPS of {state_name} is {fips}")
        print(f"BBOX of {state_name} is {bbox}")
        return fips, bbox
    else:
        fips = state_fips.get(key) if state_fips else None
        print(f"FIPS of {state_name} is {fips}")
        print(f"Warning: {state_name} has no bbox in STATE_BBOXES")
        return fips, None



def plot_top_cdl_classes(cdl, state_name, year, top_n=10, cdl_legend_lookup=CDL_LEGEND, save_dir="cdl_plots"):
    """
    Counts pixels per CDL class, maps the top-N classes using the official palette,
    and plots the overview alongside a legend panel.
    """
    # Count pixels per class and pick top-N
    counts = Counter(cdl.ravel().tolist())
    sorted_classes = sorted(counts.items(), key=lambda x: -x[1])
    top_codes = [c for c, _ in sorted_classes[:top_n]]

    # Build a uint8 display array (0 = background, 1…N = ranked classes)
    display_arr = np.zeros(cdl.shape, dtype=np.uint8)
    hex_colours, patch_labels = [], []

    for rank, code in enumerate(top_codes, start=1):
        display_arr[cdl == code] = rank
        # Use the provided legend lookup mapping
        name, hx = cdl_legend_lookup.get(code, (f"Class {code}", "#888888"))
        hex_colours.append(hx)
        
        pct = 100.0 * counts[code] / cdl.size
        patch_labels.append(f"{name}  ({code})   {pct:.1f}%")

    cmap = ListedColormap(["#000000"] + hex_colours)
    norm = BoundaryNorm(list(range(top_n + 2)), cmap.N)

    fig, (ax_map, ax_leg) = plt.subplots(
        1, 2, figsize=(14, 6),
        gridspec_kw={"width_ratios": [2, 1]}
    )

    ax_map.imshow(display_arr, cmap=cmap, norm=norm, interpolation="nearest")
    # Dynamic title string
    ax_map.set_title(f"CDL {year} — {state_name}  (top-{top_n} classes highlighted)", fontsize=13)
    ax_map.axis("off")

    patches = [mpatches.Patch(color=hex_colours[i], label=patch_labels[i])
               for i in range(top_n)]
    ax_leg.legend(handles=patches, loc="center", fontsize=9,
                  title=f"Top {top_n} CDL classes", title_fontsize=10, frameon=False)
    ax_leg.axis("off")
    plt.tight_layout()
    
    # Save the output file to a dynamic path so multiple states/years don't overwrite
    out_file = f"{save_dir}/cdl_overview_{state_name}_{year}.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    # plt.show()
    
    print(f"Saved CDL overview plot → {out_file}")

    # Summary table
    print(f"\n{'Code':>6}  {'Class Name':<32}  {'Pixels':>12}  {'%':>6}")
    print("-" * 62)
    for code, cnt in sorted_classes[:top_n]:
        name, _ = CDL_LEGEND.get(code, (f"Class {code}", ""))
        pct = 100.0 * cnt / cdl.size
        print(f"{code:>6}  {name:<32}  {cnt:>12,}  {pct:>6.2f}")
    print("\nSaved → cdl_overview.png")

# Example Usage: 
# plot_top_cdl_classes(cdl=cdl_array, state_name="Iowa", year=2025)

def get_state_info(state_name, state_bboxes=STATE_BBOXES, state_fips=None):
    """
    Retrieve FIPS code and bounding box for a specified state name.
    """
    # Create the reverse lookup mapping
    name_to_info = {
        v["state"].upper(): (k, v["bbox"])
        for k, v in state_bboxes.items()
    }

    key = state_name.upper().strip()

    if key in name_to_info:
        fips, bbox = name_to_info[key]
        print(f"FIPS of {state_name} is {fips}")
        print(f"BBOX of {state_name} is {bbox}")
        return fips, bbox
    else:
        # Fallback for states not in STATE_BBOXES
        fips = state_fips.get(key) if state_fips else None
        print(f"FIPS of {state_name} is {fips}")
        print(f"Warning: {state_name} has no bbox in STATE_BBOXES")
        return fips, None

# Example usage:
# fips, bbox = get_state_info(state_name)


def week_start_end_from_iso(year, week):
    """
    Return Monday and Sunday date strings for an ISO week.
    Output format: YYYY.MM.DD
    """
    start_dt = date.fromisocalendar(year, week, 1)   # Monday
    end_dt = date.fromisocalendar(year, week, 7)     # Sunday
    return start_dt.strftime("%Y.%m.%d"), end_dt.strftime("%Y.%m.%d")


def build_weekly_layer_name(year, week):
    """
    Crop-CASMA weekly NDVI layer format:
    NDVI-WEEKLY_YYYY_WW_YYYY.MM.DD_YYYY.MM.DD
    """
    start_str, end_str = week_start_end_from_iso(year, week)
    return f"NDVI-WEEKLY_{year}_{week:02d}_{start_str}_{end_str}"


def download_ndvi_weekly_tiff(
    year,
    week,
    out_path,
    bbox="-52095.0,1938165.0,481755.0,2288295.0",
    width=2000,
    height=1500,
):
    layer = build_weekly_layer_name(year, week)

    url = (
        "https://nassgeo.csiss.gmu.edu/cgi-bin/mapserv"
        "?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
        f"&MAP=/DATA/SMAP_DATA/WMS/NDVI-WEEKLY_{year}.map"
        f"&LAYERS={layer}"
        "&CRS=EPSG:5070"
        "&FORMAT=image/tiff"
        f"&BBOX={bbox}"
        f"&WIDTH={width}&HEIGHT={height}"
    )

    r = requests.get(url, timeout=120)
    ct = r.headers.get("Content-Type", "")
    print(f"  {layer}  →  HTTP {r.status_code}  ({ct})")

    if "tiff" in ct.lower():
        with open(out_path, "wb") as f:
            f.write(r.content)
        return True
    else:
        print("  ERROR — response:", r.text[:500])
        return False
# def reproject_to_match(src_array, src_profile, dst_profile):
#     """Reproject src_array onto dst_profile's grid using bilinear resampling."""
#     dst = np.empty((dst_profile["height"], dst_profile["width"]), dtype=np.float32)
#     reproject(
#         source=src_array.astype(np.float32),
#         destination=dst,
#         src_transform=src_profile["transform"],
#         src_crs=src_profile["crs"],
#         dst_transform=dst_profile["transform"],
#         dst_crs=dst_profile["crs"],
#         resampling=Resampling.bilinear,
#     )
#     return dst

def reproject_to_match(src_array, src_profile, dst_profile):
    dst = np.full(
        (dst_profile["height"], dst_profile["width"]),
        fill_value=np.nan,
        dtype=np.float32
    )
    reproject(
        source=src_array,
        destination=dst,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        dst_transform=dst_profile["transform"],
        dst_crs=dst_profile["crs"],
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return dst

def get_growing_season_weeks(year, start_month=3, start_day=1, end_month=10, end_day=10):
    """
    Generate ISO year/week pairs covering the growing season.

    Default season:
        March 1 -> October 10

    Returns
    -------
    list[tuple[int, int]]
        [(iso_year, iso_week), ...]
    """
    start_dt = date(year, start_month, start_day)
    end_dt = date(year, end_month, end_day)

    seen = set()
    weeks = []

    current = start_dt
    while current <= end_dt:
        iso_year, iso_week, _ = current.isocalendar()
        if (iso_year, iso_week) not in seen:
            seen.add((iso_year, iso_week))
            weeks.append((iso_year, iso_week))
        current += timedelta(days=7)

    print("Growing season weeks:")
    for iso_year, iso_week in weeks:
        print(f"  {iso_year} week {iso_week:02d}")

    return weeks


def download_growing_season_weekly_ndvi(
    year,
    out_dir,
    bbox="-52095.0,1938165.0,481755.0,2288295.0",
    width=2000,
    height=1500,
    mapfile_year=2025,
    layer_prefix="NDVI-WEEKLY",
):
    """
    Download all weekly NDVI layers for the growing season.
    """
    os.makedirs(out_dir, exist_ok=True)

    weeks = get_growing_season_weeks(year)
    downloaded = []

    for iso_year, iso_week in weeks:
        out_name = f"{layer_prefix}_{iso_year}_{iso_week:02d}.tif"
        out_path = os.path.join(out_dir, out_name)

        ok = download_ndvi_weekly_tiff(
            year=iso_year,
            week=iso_week,
            out_path=out_path,
            bbox=bbox,
            width=width,
            height=height,
            mapfile_year=mapfile_year,
            layer_prefix=layer_prefix,
        )

        if ok:
            downloaded.append(out_path)

    print(f"\nDownloaded {len(downloaded)} weekly files.")
    return downloaded

def build_smap_weekly_layer(year, week):
    """
    SMAP weekly anomaly layer name format:
    SMAP-9KM-ANOMALY-WEEKLY-SUB_YYYY_WW_YYYY.MM.DD_YYYY.MM.DD
    """
    start_str, end_str = week_start_end_from_iso(year, week)
    return f"SMAP-9KM-ANOMALY-WEEKLY-SUB_{year}_{week:02d}_{start_str}_{end_str}"


def download_smap_weekly(year, week, output_file,
                          bbox='-54000,1929000,471000,2293000', crs="http://www.opengis.net/def/crs/EPSG/0/5070"):
    """Download one weekly SMAP anomaly GeoTIFF."""
    if os.path.exists(output_file):
        print(f"  [SMAP {year} w{week:02d}] Already exists, skipping.")
        return True

    layer = build_smap_weekly_layer(year, week)

    params = {
        'SERVICE':      'WCS',
        'VERSION':      '2.0.1',
        'REQUEST':      'GetCoverage',
        'MAP':          f'/WMS/SMAP-9KM-ANOMALY-WEEKLY-SUB_{year}.map',
        'COVERAGEID':   layer,
        'FORMAT':       'image/tiff',
        'SUBSET':       [
            f'x({bbox.split(",")[0]},{bbox.split(",")[2]})',
            f'y({bbox.split(",")[1]},{bbox.split(",")[3]})',
        ],
        'SUBSETTINGCRS': crs,
    }

    print(f"  Downloading {layer} ...")
    try:
        r = requests.get(
            "https://cloud.csiss.gmu.edu/smap_server/cgi-bin/mapserv",
            params=params, timeout=60
        )
        ct = r.headers.get("Content-Type", "")
        if r.status_code == 200 and "tiff" in ct.lower():
            with open(output_file, "wb") as f:
                f.write(r.content)
            print(f"  Saved → {output_file}")
            return True
        else:
            print(f"  Error {r.status_code} | {ct} | {r.text[:300]}")
            return False
    except Exception as e:
        print(f"  Exception: {e}")
        return False


def download_smap_growing_season(year, out_dir="smap_weekly",
                                  start_month=3, end_month=10, bbox='-54000,1929000,471000,2293000'):
    """Download all weekly SMAP anomaly layers for the growing season."""
    os.makedirs(out_dir, exist_ok=True)

    # Generate ISO weeks covering the growing season
    start_dt = date(year, start_month, 1)
    end_dt   = date(year, end_month, 10)

    seen, weeks = set(), []
    cur = start_dt
    while cur <= end_dt:
        yw = cur.isocalendar()[:2]  # (iso_year, iso_week)
        if yw not in seen:
            seen.add(yw)
            weeks.append(yw)
        cur += timedelta(days=7)

    print(f"Downloading {len(weeks)} weekly SMAP layers for {year}...")
    downloaded = []
    for iso_year, iso_week in weeks:
        fname = f"smap_anomaly_{iso_year}_w{iso_week:02d}.tif"
        out_path = os.path.join(out_dir, fname)
        ok = download_smap_weekly(iso_year, iso_week, out_path, bbox=bbox)
        if ok:
            downloaded.append(out_path)

    print(f"\nDone — {len(downloaded)}/{len(weeks)} files downloaded.")
    return downloaded


# ============================================================================
# Task 4 Lightweight Helper Functions
# ============================================================================

def download_smap_anomaly(date_str, output_file, bbox=None, crs="http://www.opengis.net/def/crs/EPSG/0/5070"):
    """
    Download SMAP L4 soil moisture anomaly data for a specific date.
    
    Parameters
    ----------
    date_str : str
        Date string in format YYYY.MM.DD
    output_file : str
        Path to save the downloaded TIFF file
    bbox : str
        Bounding box string in format "xmin,ymin,xmax,ymax"
    crs : str
        Coordinate reference system URL
    
    Returns
    -------
    bool
        True if download succeeded, False otherwise
    """
    year = date_str.split('.')[0]
    url = "https://cloud.csiss.gmu.edu/smap_server/cgi-bin/mapserv"
    params = {
        'SERVICE': 'WCS',
        'VERSION': '2.0.1',
        'REQUEST': 'GetCoverage',
        'MAP': f'/WMS/SMAP-9KM-ANOMALY-DAILY-SUB_{year}.map',
        'COVERAGEID': f'SMAP-9KM-ANOMALY-DAILY-SUB_{date_str}',
        'FORMAT': 'image/tiff',
        'SUBSET': [f'x({bbox.split(",")[0]},{bbox.split(",")[2]})', 
                   f'y({bbox.split(",")[1]},{bbox.split(",")[3]})'],
        'SUBSETTINGCRS': crs
    }
    
    print(f"Downloading SMAP Anomaly for {date_str}...")
    r = requests.get(url, params=params, timeout=60)
    if r.status_code == 200:
        with open(output_file, 'wb') as f:
            f.write(r.content)
        print(f"  Saved to {output_file}")
        return True
    else:
        print(f"  Error: {r.status_code}")
        print(r.text[:500])
        return False

def task4_build_ndvi_lightweight_stack(year, state_name, cdl_profile, out_dir="ndvi_data_task4"):
    """
    Lightweight NDVI stack for Task 4: download only 3 weekly layers per year.
    
    Downloads exactly 3 representative weekly NDVI layers:
    - Early: One April week (week ~15)
    - Peak: One July week (week ~28)
    - Late: One September week (week ~37)
    
    Returns a 3-band stack (early, peak, late) aligned to CDL grid.
    
    Parameters
    ----------
    year : int
        Year to download for
    state_name : str
        State name for file naming
    cdl_profile : dict
        CDL raster profile for reprojection
    out_dir : str
        Output directory for downloaded files
    
    Returns
    -------
    numpy.ndarray
        3-band stack with shape (3, height, width)
        Band 0: early season (April)
        Band 1: peak season (July)
        Band 2: late season (September)
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Define 3 representative weeks (one per season)
    early_week = 15   # April
    peak_week = 28    # July
    late_week = 37    # September
    
    weeks_to_download = [(year, early_week), (year, peak_week), (year, late_week)]
    
    bands = []
    for iso_year, iso_week in weeks_to_download:
        fname = f"{out_dir}/ndvi_task4_{state_name}_{iso_year}_w{iso_week:02d}.tif"
        
        # Download if not exists
        if not os.path.exists(fname):
            print(f"  Downloading NDVI week {iso_week} for {year}...")
            ok = download_ndvi_weekly_tiff(iso_year, iso_week, fname)
            if not ok:
                print(f"    Failed to download week {iso_week}, using NaN")
                bands.append(np.full((cdl_profile["height"], cdl_profile["width"]), np.nan))
                continue
        
        # Load and reproject
        try:
            with rasterio.open(fname) as src:
                ndvi_raw = src.read(1).astype(np.float32)
                ndvi_profile = src.profile
                nodata_val = src.nodata
                
                if nodata_val is not None:
                    ndvi_raw[ndvi_raw == nodata_val] = np.nan
                
                # Scale uint8 -> float NDVI
                ndvi_float = ndvi_raw / 125.0 - 1.0
                ndvi_matched = reproject_to_match(ndvi_float, ndvi_profile, cdl_profile)
                ndvi_matched[(ndvi_matched < -1) | (ndvi_matched > 1)] = np.nan
                
                bands.append(ndvi_matched)
                print(f"    ✓ Loaded NDVI week {iso_week}")
        except Exception as e:
            print(f"    Error loading NDVI week {iso_week}: {e}")
            bands.append(np.full((cdl_profile["height"], cdl_profile["width"]), np.nan))
    
    # Stack into 3-band array
    stack = np.stack(bands)
    
    return stack


def task4_build_smap_lightweight_stack(year, state_name, cdl_profile, bbox, out_dir="smap_data_task4"):
    """
    Lightweight SMAP stack for Task 4: download only 3 dates per year.
    
    Downloads exactly 3 representative SMAP anomaly dates:
    - Early: April 15 (YYYY.04.15)
    - Mid: July 15 (YYYY.07.15)
    - Late: September 15 (YYYY.09.15)
    
    Returns a 3-band stack (early, mid, late) aligned to CDL grid.
    
    Parameters
    ----------
    year : int
        Year to download for
    state_name : str
        State name for file naming
    cdl_profile : dict
        CDL raster profile for reprojection
    bbox : str
        Bounding box string in format "xmin,ymin,xmax,ymax"
    out_dir : str
        Output directory for downloaded files
    
    Returns
    -------
    numpy.ndarray
        3-band stack with shape (3, height, width)
        Band 0: early season (April 15)
        Band 1: mid season (July 15)
        Band 2: late season (September 15)
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Define 3 representative dates
    dates_to_download = [
        f"{year}.04.15",  # Early
        f"{year}.07.15",  # Mid
        f"{year}.09.15",  # Late
    ]
    
    bands = []
    for date_str in dates_to_download:
        fname = f"{out_dir}/smap_task4_{state_name}_{date_str.replace('.', '')}.tif"
        
        # Download if not exists
        if not os.path.exists(fname):
            print(f"  Downloading SMAP for {date_str}...")
            ok = download_smap_anomaly(date_str, fname, bbox=bbox)
            if not ok:
                print(f"    Failed to download {date_str}, using NaN")
                bands.append(np.full((cdl_profile["height"], cdl_profile["width"]), np.nan))
                continue
        
        # Load and reproject
        try:
            with rasterio.open(fname) as src:
                sm_raw = src.read(1).astype(np.float32)
                sm_profile = src.profile
                nodata_val = src.nodata
                
                if nodata_val is not None:
                    sm_raw[sm_raw == nodata_val] = np.nan
                
                sm_matched = reproject_to_match(sm_raw, sm_profile, cdl_profile)
                
                bands.append(sm_matched)
                print(f"    ✓ Loaded SMAP for {date_str}")
        except Exception as e:
            print(f"    Error loading SMAP for {date_str}: {e}")
            bands.append(np.full((cdl_profile["height"], cdl_profile["width"]), np.nan))
    
    # Stack into 3-band array
    stack = np.stack(bands)
    
    return stack


def task4_extract_ndvi_last_year(ndvi_stacks, target_year, rows, cols):
    """
    Extract NDVI features from the previous year (t-1) only for Task 4.
    
    Parameters
    ----------
    ndvi_stacks : dict
        Dictionary mapping year to 3-band NDVI stack (early, peak, late)
        Each stack shape: (3, height, width)
    target_year : int
        Target year (will use target_year - 1)
    rows, cols : array-like
        Pixel indices to extract values for
    
    Returns
    -------
    dict
        Dictionary with last-year NDVI features:
        - early_ndvi_last_year
        - peak_ndvi_last_year
        - late_ndvi_last_year
    """
    last_year = target_year - 1
    
    if last_year not in ndvi_stacks:
        # Return NaN if last year data not available
        return {
            'early_ndvi_last_year': np.full(len(rows), np.nan),
            'peak_ndvi_last_year': np.full(len(rows), np.nan),
            'late_ndvi_last_year': np.full(len(rows), np.nan),
        }
    
    stack = ndvi_stacks[last_year]
    # stack shape: (3, height, width)
    # band 0: early, band 1: peak, band 2: late
    
    return {
        'early_ndvi_last_year': stack[0, rows, cols],
        'peak_ndvi_last_year': stack[1, rows, cols],
        'late_ndvi_last_year': stack[2, rows, cols],
    }


def task4_extract_smap_last_year(smap_stacks, target_year, rows, cols):
    """
    Extract SMAP features from the previous year (t-1) only for Task 4.
    
    Parameters
    ----------
    smap_stacks : dict
        Dictionary mapping year to 3-band SMAP stack (early, mid, late)
        Each stack shape: (3, height, width)
    target_year : int
        Target year (will use target_year - 1)
    rows, cols : array-like
        Pixel indices to extract values for
    
    Returns
    -------
    dict
        Dictionary with last-year SMAP features:
        - sm_early_last_year
        - sm_mid_last_year
        - sm_late_last_year
    """
    last_year = target_year - 1
    
    if last_year not in smap_stacks:
        # Return NaN if last year data not available
        return {
            'sm_early_last_year': np.full(len(rows), np.nan),
            'sm_mid_last_year': np.full(len(rows), np.nan),
            'sm_late_last_year': np.full(len(rows), np.nan),
        }
    
    stack = smap_stacks[last_year]
    # stack shape: (3, height, width)
    # band 0: early, band 1: mid, band 2: late
    
    return {
        'sm_early_last_year': stack[0, rows, cols],
        'sm_mid_last_year': stack[1, rows, cols],
        'sm_late_last_year': stack[2, rows, cols],
    }
