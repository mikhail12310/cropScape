# CropScape & NASA SMAP Agricultural Analytics

This repository operates an end-to-end data engineering and agricultural data science pipeline. It dynamically retrieves multi-terabyte spatial telemetry from the USDA CropScape and NASA SMAP remote sensing clusters and translates them into machine-learning frameworks built to predict crop phenotypes and assess extreme weather events.

## Prerequisites & Setup

1. **Environment Sandbox:**
   A local virtual environment is highly recommended to prevent namespace crashes with your system-level binaries. 
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Library Dependencies:** The core analytical matrix relies heavily on `rasterio`, `numpy`, `pandas`, and `matplotlib`.
3. **The Core Engine (`utils.py`):** Do not delete or move `utils.py`. It holds all standardized API scrapers, GMU CropSmart WCS/WMS connection credentials, ISO week string formatters, and CRS reprojection mathematics that power the Jupyter components.

---

## Step-By-Step Execution Flow

### Task 1: Automated Spatial Acquisition (`Task1_Master_clean.ipynb`)
**Goal:** Scrape local Cropland Data Layers (CDL) and NDVI subsets directly from the official government servers, constrained exactly to user-defined state limits.
**Execution Instructions:**
1. Open the internal `task1_config.json` file to establish your target matrix. You can define specific target arrays here, such as: `{"states": ["Iowa", "Illinois"], "years": [2022, 2025]}`.
2. Run all cells in `Task1_Master_clean.ipynb`.
3. **Output:** The notebook automatically executes the spatial queries, downloads the `.tif` arrays, isolates the dominant local crop types (primarily Corn and Soybeans), and spits out visualization arrays into `/cdl_plots/` and tabular crop accounting to `/summary/`.

### Task 2: Crop Rotation & History Analysis (`Task2_Crop_Rotation.ipynb`)
**Goal:** Exploit multi-year USDA CDL records to isolate local agronomic rotation behaviours.
**Execution Instructions:**
1. Task 1 must be completed first to build the requisite spatial `.tif` library in memory. 
2. Open `Task2_Crop_Rotation.ipynb` and trigger it.
3. **Output:** The script will align historical planting sequences pixel-by-pixel (tracking dominant Corn-Soy-Corn schedules) and identify instances of irregular sequences or rotation breaks that serve as a predictive baseline filter for Task 4.

### Task 3: Drought Deficit Analysis (`Task3_SMAP_Analysis.ipynb`)
**Goal:** Track the extreme sub-soil moisture starvation patterns associated with the 2022 Midwest Flash Drought and map them directly against active biological crop-states.
**Execution Instructions:**
1. Run the entire `Task3_SMAP_Analysis.ipynb` notebook.
2. The notebook will automatically bypass CropSmart UI constraints, looping through an entire 5-year historical timeframe (2017-2021) to calculate physical baseline climatology, before dynamically querying and mathematically processing the 2022 stress anomaly vector.
3. **Output:** Generating a robust 4-panel dashboard containing time-series scalars tracking the average m³/m³ deficit specifically mapped *only* over known Iowa Corn and Soybean footprints, displaying exactly when extreme stress crashed into critical tasseleling stages. See the `Final_Report_Answers.md` for a full write-up on this dynamic.

### Task 4: Machine Learning Predictor (`Task4_Professional_Pipeline.ipynb`)
**Goal:** Execute a multi-feature Machine Learning architecture (Random Forest / XGBoost classifiers) configured to predict in-season crop types by synthesizing temporal NDVI phenotypes and historical rotation biases.
**Execution Instructions:**
1. Run all cells within `Task4_Professional_Pipeline.ipynb`.
2. The notebook effortlessly unpackages the pre-compiled `.pkl` binary training/testing arrays loaded via `git pull` without requiring manual recalculation.
3. **Output:** The model will execute its scoring runs, outputting local variable importance structures (`prof_backbone_importance.png`), determining exactly how early in the growing season the algorithm can confidently differentiate early-stage Corn emergence from Soybeans based entirely on orbital telemetry parameters.

---

## ⚠️ Notes & Troubleshooting
* **CRS Integrity:** Ensure all generated rasters match the global `EPSG:5070` protocol. Bounding Box mismatches will trigger catastrophic `rasterio` index boundaries.
* **Server Throttling:** If WCS Server endpoints drop under heavy load during Task 1 or 3 downloads, the system is engineered to gracefully throw `np.nan` blocks and bypass the week without nuking downstream aggregation formulas.