# 🗂️ EY Urban Heat Island Challenge – Walk-through
Below is a step-by-step guide that walks through the whole project lifecycle, from setting up the environment to delivering a final model. It assumes no prior knowledge of the repository, so everything is explained from the ground up.

---

## 1️⃣ Project Overview

| Item | Details |
| :--- | :--- |
| **Goal** | Build a **classification** model that predicts the **Urban Heat Island (UHI) intensity** (Low / Medium / High) for a new city (Sierra Leone) using ground-truth temperature points, Sentinel-2 satellite imagery, and building-footprint data. |
| **Inputs** | • `sample_brazil_uhi_data.csv` & `sample_chile_uhi_data.csv` – latitude, longitude, UHI class (training) <br> • `Validation_Dataset.csv` – locations to predict (test) <br> • Sentinel-2 GeoTIFFs (spectral bands & indices) <br> • Building-footprint shapefiles (height, density) |
| **Outputs** | • `submission.csv` with predicted UHI class for every validation point <br> • Optional model artefacts (trained model, feature importance, visualisations) |
| **Success Metric** | Classification **accuracy / macro-F1** on a hidden leaderboard (the challenge uses a hidden test set). Aim for **≥ 0.70 macro-F1** as a solid baseline. |

---

## 2️⃣ Set-up the Development Environment

| Step | Command | Explanation |
| :--- | :--- | :--- |
| **2.1 Clone the repo** | `git clone <repo-url>` | Bring the code & data into your local folder. |
| **2.2 Create a virtual environment** | `python -m venv .venv` <br> `source .venv/bin/activate` (Linux/macOS) <br> `.\.venv\Scripts\activate` (Windows) | Isolates dependencies from your system Python. |
| **2.3 Install required packages** | `pip install -r requirements.txt` | Installs all libraries used in the starter notebook (e.g., **numpy**, **pandas**, **scikit-learn**, **rasterio**, **geopandas**, **matplotlib**, **seaborn**). |
| **2.4 Optional – JupyterLab** | `pip install jupyterlab` <br> `jupyter lab` | Opens an interactive notebook environment (the starter notebook is `Sample_Model_Notebook.ipynb`). |
| **2.5 Verify the install** | Run `python -c "import rasterio, geopandas, sklearn; print('OK')"` | Confirms that heavy GIS libraries load correctly. |

> **Tip:** Keep the environment reproducible by exporting it (`pip freeze > requirements.txt`) after you add any new package.

---

## 3️⃣ Understand the Data

| Dataset | What it contains | Key things to check |
| :--- | :--- | :--- |
| `sample_brazil_uhi_data.csv` & `sample_chile_uhi_data.csv` | Latitude, longitude, UHI class (Low/Medium/High). | • Count of points per class (class imbalance). <br> • Geographic spread (visualise on a map). |
| `Validation_Dataset.csv` | Only lat/lon (no label). | • Same coordinate system as training data (WGS84). |
| Sentinel-2 GeoTIFFs (*.tif) | 10 m optical bands + derived indices (NDVI, NDBI, etc.) for each region. | • Cloud-free vs. cloudy scenes (use the provided cloud-mask notebook). <br> • Band order & scaling (Sentinel-2 uses reflectance × 10,000). |
| Building-footprint shapefiles (*.shp) | Polygon geometry + height/area attributes. | • CRS (usually EPSG:4326 or a local projection). <br> • Missing attributes (fill with median height if needed). |

> **Quick sanity check:** Load a few rows in Python and plot them on a basemap (e.g., `geopandas` + `contextily`) to ensure coordinates line up with the satellite imagery.

---

## 4️⃣ Exploratory Data Analysis (EDA)

* **Spatial visualisation** – Plot training points over the Sentinel-2 mosaic to see where “hot” vs. “cold” points sit.
* **Class distribution** – `train['UHI_class'].value_counts()` → note any imbalance.
* **Correlation heatmap** – Between spectral bands / indices and the target. Use `seaborn.heatmap`.
* **Feature sanity** – Compute simple statistics for building density (e.g., number of buildings per km²) and height.
* **Outlier detection** – Extreme NDVI or NDBI values may indicate water bodies or shadows; consider masking them.

**Deliverable:** A short notebook (`EDA.ipynb`) with plots saved as PNGs and a markdown summary of findings.

---

## 5️⃣ Feature Engineering

| Feature | How to create | Why it matters |
| :--- | :--- | :--- |
| **Spectral band values** | Sample the raster at each point using `rasterio.sample`. | Direct surface reflectance. |
| **NDVI** (Normalized Difference Vegetation Index) | (NIR - RED) / (NIR + RED) | Vegetation cools the surface → lower UHI. |
| **NDBI** (Normalized Difference Built-up Index) | (SWIR - NIR) / (SWIR + NIR) | Built-up density → higher UHI. |
| **Mean/Std of bands in a 30 m buffer** | `geopandas.buffer(30)` → zonal stats (`rasterstats`). | Captures neighbourhood context. |
| **Building density** | Count footprints intersecting a 100 m buffer / area. | More buildings → higher heat. |
| **Average building height** | Mean of `height` attribute in buffer. | Tall buildings trap heat. |
| **Distance to water** | Euclidean distance to nearest water polygon. | Proximity to water reduces heat. |
| **Temporal features** (optional) | Day of year, hour of acquisition (if multiple dates). | Seasonal effects. |

> **Implementation tip:** Write a reusable function `extract_features(df, raster_paths, footprint_gdf)` that returns a `pandas.DataFrame` ready for modelling. Store the result as `features_train.parquet` for quick re-use.

---

## 6️⃣ Modeling

### 6.1 Baseline Model
* **Algorithm:** `RandomForestClassifier` (good default, handles mixed data).
* **Parameters:** `n_estimators=300`, `max_depth=None`, `class_weight='balanced'`.
* **Cross-validation:** 5-fold stratified CV on the combined Brazil + Chile data.

### 6.2 Advanced Options
| Model | When to use | Key hyper-parameters |
| :--- | :--- | :--- |
| **XGBoost / LightGBM** | Need higher performance, can handle missing values. | `learning_rate`, `max_depth`, `n_estimators`. |
| **Gradient Boosted Trees (sklearn)** | Simpler API, still strong. | Same as RF but with `learning_rate`. |
| **Neural Net (PyTorch / TensorFlow)** | Large training set, want to learn spatial patterns directly. | Small CNN on raster patches + building-mask channel. |
| **Stacking / Ensemble** | Combine strengths of multiple models. | Use `sklearn.ensemble.StackingClassifier`. |

### 6.3 Evaluation Metrics
* **Macro-F1** – primary competition metric.
* **Confusion matrix** – see which classes are confused.
* **Feature importance** – `model.feature_importances_` (RF/XGB) → sanity check.

### 6.4 Model Persistence
```python
import joblib
joblib.dump(model, "models/rf_uhi.pkl")
```
Store the model under models/ and version-control the script that creates it (train\_model.py).

---

## 7️⃣ Prediction & Submission

1. **Load validation points** (`Validation_Dataset.csv`).  
2. **Extract the same features** as for training (use the same function).  
3. **Predict** with the trained model: `pred = model.predict(features_val`).  
4. **Create submission file:**

```csv  
id,UHI_class  
0,Low  
1,Medium  
```
5. **Save** as `submission.csv` in the repo root (or a `submissions/` folder with a timestamp).

> **Tip:** Keep a `README_submissions.md` that logs the model version, hyper-parameters, and validation score for each submission.

---

## 8️⃣ Best Practices & Project Hygiene

| Area | Guideline |
| :--- | :--- |
| **Folder Structure** | *check below this table* |
| **Version Control** | • Commit early & often. <br> • Use **feature branches** (`feature/eda`, `feature/rf-model`). <br> • Write clear commit messages (`git commit -m "Add NDVI feature extraction"`). |
| **Reproducibility** | • Pin exact library versions (`pip freeze > requirements.txt`). <br> • Store random seeds (`np.random.seed(42)`). |
| **Documentation** | • Keep the top-level `README.md` up-to-date with a project summary, folder map, and run instructions. <br> • Add docstrings to every function (type hints help!). |
| **Coding Style** | • Follow **PEP-8** (use `black` or `ruff` for auto-format). <br> • Use **meaningful variable names** (`ndvi`, `building_density`). |
| **Testing** | • Write a few **unit tests** (`pytest`) for the feature extraction (e.g., check that a point inside a building gets a non-zero density). |
| **Data Handling** | • Never commit raw GeoTIFFs larger than 100 MB – add them to `.gitignore` and provide a download script (`download_data.sh`). <br> • Use **Parquet** for intermediate feature tables (fast I/O). |
| **Experiment Tracking** | • Log metrics with a lightweight tool (e.g., `mlflow` or simple CSV `experiment_log.csv`). |
| **Collaboration** | • Use **pull requests** for code review. <br> • Tag teammates for review (`@username`). |
| **Ethics & Privacy** | • Ensure no personally-identifiable data is shared. <br> • Cite the original datasets (see the References section in the README). |

The folder structure:
```text
EY_UHI_Challenge/
├── data/                 # Raw and processed datasets
├── notebooks/            # EDA, baseline models, and experiments
├── src/                  # Source code
│   ├── feature_engineering.py
│   ├── train.py
│   └── predict.py
├── models/               # Saved model files (.pkl)
├── submissions/          # Timestamped submission CSVs
├── reports/              # Generated plots and markdown summaries
└── requirements.txt      # Project dependencies
```
---

## 9️⃣ Suggested Scripts (Skeleton)

| Script | Purpose | Key Functions |
| :---- | :---- | :---- |
| `src/feature_engineering.py` | Extract all features from rasters & shapefiles. | `load_rasters()`, `sample_point()`, `compute_ndvi()`, `zonal_stats()`. |
| `src/train.py` | Train a model given a feature CSV. | `load_features()`, ``split_train_val()``, `train_rf()`, `save_model()`. |
| `src/predict.py` | Generate predictions for the validation set. | `load_model()`, `extract_features()`, `predict()`, `write_submission()`. |
| `src/evaluate.py` | Compute CV scores & plot confusion matrix. | `cross_val_score()`, `plot_confusion()`. |
| `download_data.sh` | Helper script to pull large GeoTIFFs from cloud storage. | `az storage blob download …`. |

> All scripts should accept **command-line arguments** (e.g., \--input data/train\_features.parquet) so they can be run from the terminal or a notebook.

---

## 🔟 Putting It All Together – A Typical Workflow
```text
1. git checkout -b feature/baseline-rf  
2. source .venv/bin/activate  
3. python src/feature_engineering.py --region brazil # creates data/brazil_features.parquet  
4. python src/feature_engineering.py --region chile # creates data/chile_features.parquet  
5. python src/train.py --input data/brazil_features.parquet data/chile_features.parquet --model-out models/rf_uhi.pkl --log experiment_log.csv  
6. python src/predict.py --model models/rf_uhi.pkl --validation data/Validation_Dataset.csv --submission submissions/2024-09-01_rf.csv  
7. git add .  
8. git commit -m "Baseline RandomForest model (macro-F1 = 0.71)"  
9. git push origin feature/baseline-rf  
10. Open a PR, request review, merge to main.
```

## 📚 Further Reading & Resources

| Topic | Link / Reference |
| :---- | :---- |
| **Sentinel-2 data handling** | ESA Sentinel-2 Toolbox, `rasterio` docs |
| **Geospatial feature extraction** | `rasterstats` (zonal statistics) |
| **Imbalanced classification** | Scikit-learn’s `class_weight='balanced'`, SMOTE |
| **Model interpretability** | SHAP values for tree models |
| **Competition best-practices** | Kaggle “How to win a competition” posts |

---

## **🎉 Final Words**

* **Start simple.** Get a working RandomForest baseline before diving into deep learning. Check the `Sample_Model_Notebook.ipynb`  
* **Iterate fast.** Each new feature should be validated with a quick CV run.  
* **Document everything.** Future you (or a teammate) will thank you when the project scales.

Good luck, and enjoy turning satellite data into actionable insights on urban heat islands\! 🚀

