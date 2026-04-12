import os
import time
import re
import requests

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