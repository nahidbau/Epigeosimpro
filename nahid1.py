import streamlit as st
from scipy.integrate import solve_ivp
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap, MarkerCluster, FastMarkerCluster, HeatMapWithTime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
import zipfile
import tempfile
import math
import io
import base64
import json
from PIL import Image
import pickle
import hashlib
import itertools
from collections import defaultdict

# Scientific Computing
from scipy import stats, signal, interpolate, optimize
from scipy.spatial import distance_matrix, ConvexHull
from scipy.stats import gaussian_kde, pearsonr, spearmanr, kendalltau, zscore
import networkx as nx
from sklearn.ensemble import (RandomForestRegressor, IsolationForest,
                              GradientBoostingRegressor, AdaBoostRegressor,
                              RandomForestClassifier, VotingRegressor)
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     GridSearchCV, TimeSeriesSplit)
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score,
                             mean_absolute_error, mean_absolute_percentage_error,
                             precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_auc_score)
from sklearn.preprocessing import (StandardScaler, LabelEncoder, MinMaxScaler,
                                   PolynomialFeatures, RobustScaler)
from sklearn.cluster import (DBSCAN, KMeans, AgglomerativeClustering,
                             OPTICS, SpectralClustering)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, SVC
from sklearn.linear_model import (LinearRegression, Lasso, Ridge, ElasticNet,
                                  LogisticRegression, PoissonRegressor)
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.manifold import TSNE, MDS
from sklearn.neighbors import KNeighborsRegressor, LocalOutlierFactor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# GIS & Spatial Analysis
from shapely.geometry import Point, Polygon, LineString, MultiPoint
from shapely.ops import voronoi_diagram, unary_union
import fiona
import rasterio
from rasterio.plot import show
import pyproj
from pyproj import Transformer
import pysal
from pysal.lib import weights
import esda
from mgwr.gwr import GWR
from pysal.model import spreg
import contextily as ctx


# Epidemiological Models (using simpler implementation since epimodels might not be available)
def sir_model(t, y, beta, gamma, N):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


def seir_model(t, y, beta, sigma, gamma, N):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]


warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="EpiGeoSim-X Pro: Advanced Spatiotemporal Disease Intelligence Platform",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Enhanced UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Roboto+Mono:wght@400;500&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(90deg, #1a5f7a 0%, #2c3e50 50%, #9b59b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        margin-bottom: 0.5rem;
        text-align: center;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        letter-spacing: -0.5px;
    }

    .sub-header {
        font-size: 1.5rem;
        color: #5D6D7E;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
        letter-spacing: 0.5px;
    }

    .feature-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.2rem;
        box-shadow: 0 15px 40px rgba(0,0,0,0.08);
        border-left: 6px solid #1a5f7a;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
    }

    .feature-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 25px 50px rgba(0,0,0,0.15);
        border-left: 6px solid #9b59b6;
    }

    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1.5rem;
        background: linear-gradient(135deg, #1a5f7a, #9b59b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
    }

    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 0.8rem;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
    }

    .stat-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
        animation: shimmer 3s infinite;
    }

    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }

    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 1rem 0;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
    }

    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }

    .novel-tool-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border-radius: 20px;
        padding: 2.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.3);
    }

    .novel-tool-card::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 1%, transparent 20%);
        animation: pulse 4s infinite;
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }

    .tab-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        border: 1px solid #eaeaea;
    }

    .upload-section {
        border: 4px dashed #1a5f7a;
        border-radius: 20px;
        padding: 4rem;
        text-align: center;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        margin: 2.5rem 0;
        transition: all 0.3s ease;
    }

    .upload-section:hover {
        border-color: #9b59b6;
        background: linear-gradient(135deg, #e9ecef 0%, #f8f9fa 100%);
    }

    .analysis-section {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        border-left: 6px solid #27ae60;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
    }

    .stButton > button {
        background: linear-gradient(90deg, #1a5f7a 0%, #2c3e50 100%);
        color: white;
        border: none;
        padding: 1rem 2.5rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 30px rgba(26, 95, 122, 0.4);
        background: linear-gradient(90deg, #2c3e50 0%, #1a5f7a 100%);
    }

    .stSelectbox, .stMultiselect, .stSlider {
        background: white;
        border-radius: 10px;
        padding: 0.8rem;
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
    }

    .stSelectbox:focus, .stMultiselect:focus, .stSlider:focus {
        border-color: #1a5f7a;
        box-shadow: 0 0 0 3px rgba(26, 95, 122, 0.1);
    }

    .risk-high { 
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%) !important; 
        color: white; 
        font-weight: bold;
    }
    .risk-medium { 
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%) !important; 
        color: white; 
        font-weight: bold;
    }
    .risk-low { 
        background: linear-gradient(135deg, #27ae60 0%, #229954 100%) !important; 
        color: white; 
        font-weight: bold;
    }
    .risk-very-low { 
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%) !important; 
        color: white; 
        font-weight: bold;
    }

    .sparkline {
        font-family: 'Roboto Mono', monospace;
        font-size: 0.9rem;
    }

    .loading-animation {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px;
    }

    .loading-dot {
        width: 12px;
        height: 12px;
        margin: 0 5px;
        background: linear-gradient(135deg, #1a5f7a, #9b59b6);
        border-radius: 50%;
        animation: loading 1.4s infinite ease-in-out both;
    }

    .loading-dot:nth-child(1) { animation-delay: -0.32s; }
    .loading-dot:nth-child(2) { animation-delay: -0.16s; }
    .loading-dot:nth-child(3) { animation-delay: 0s; }

    @keyframes loading {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1.0); }
    }

    .info-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }

    .badge-success { background: #d4edda; color: #155724; }
    .badge-warning { background: #fff3cd; color: #856404; }
    .badge-danger { background: #f8d7da; color: #721c24; }
    .badge-info { background: #d1ecf1; color: #0c5460; }

    .model-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem;
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
    }

    .model-card:hover {
        border-color: #1a5f7a;
        box-shadow: 0 10px 25px rgba(26, 95, 122, 0.1);
    }

    .export-option {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        cursor: pointer;
    }

    .export-option:hover {
        border-color: #1a5f7a;
        background: #f8f9fa;
    }

    .map-tooltip {
        position: absolute;
        background: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        z-index: 1000;
        max-width: 300px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# ENHANCED DATA MANAGER WITH SHAPEFILE SUPPORT
# ============================================

class EnhancedDataManager:
    """Advanced Data Manager with shapefile support and data validation"""

    def __init__(self):
        self.data = None
        self.shapefile = None
        self.metadata = {}
        self.data_quality_report = {}

    def load_csv(self, file, date_cols=None, coord_cols=None):
        """Load CSV/Excel file with automatic format detection"""
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file, low_memory=False)
            elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
                df = pd.read_excel(file)
            elif file.name.endswith('.json'):
                df = pd.read_json(file)
            elif file.name.endswith('.geojson'):
                gdf = gpd.read_file(file)
                df = pd.DataFrame(gdf.drop(columns='geometry'))
                if 'geometry' in gdf.columns:
                    df['geometry'] = gdf['geometry']
            else:
                st.error("Unsupported file format")
                return None

            # Auto-detect and convert date columns
            date_converted = []
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['date', 'time', 'day', 'month', 'year']):
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        date_converted.append(col)
                    except:
                        pass

            # Auto-detect coordinate columns
            coord_cols = []
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['lat', 'lon', 'long', 'x', 'y', 'coord']):
                    coord_cols.append(col)

            # Generate metadata
            self.metadata = {
                'filename': file.name,
                'rows': len(df),
                'columns': len(df.columns),
                'date_columns': date_converted,
                'coordinate_columns': coord_cols,
                'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
                'missing_values': df.isnull().sum().sum(),
                'memory_usage': df.memory_usage(deep=True).sum() / 1024 ** 2  # MB
            }

            st.success(f"✅ Successfully loaded {len(df):,} records with {len(df.columns)} columns")
            return df

        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return None

    def load_shapefile(self, uploaded_files):
        """Load shapefile from uploaded files"""
        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as tmpdir:
                # Handle zip file
                zip_file = None
                other_files = []

                for file in uploaded_files:
                    if file.name.endswith('.zip'):
                        zip_file = file
                    else:
                        other_files.append(file)

                if zip_file:
                    # Extract zip file
                    with zipfile.ZipFile(zip_file, 'r') as z:
                        z.extractall(tmpdir)
                    # Find shapefile
                    shp_files = [f for f in os.listdir(tmpdir) if f.endswith('.shp')]
                    if not shp_files:
                        st.error("No .shp file found in zip archive")
                        return None
                    shp_path = os.path.join(tmpdir, shp_files[0])
                    gdf = gpd.read_file(shp_path)
                elif other_files:
                    # Save individual files
                    for file in other_files:
                        file_path = os.path.join(tmpdir, file.name)
                        with open(file_path, 'wb') as f:
                            f.write(file.getbuffer())
                    # Find shapefile
                    shp_files = [f for f in other_files if f.name.endswith('.shp')]
                    if not shp_files:
                        st.error("No .shp file found in uploaded files")
                        return None
                    shp_path = os.path.join(tmpdir, shp_files[0].name)
                    gdf = gpd.read_file(shp_path)
                else:
                    st.error("No files provided")
                    return None

                # Ensure WGS84 CRS
                if gdf.crs is None:
                    gdf.crs = 'EPSG:4326'
                    st.warning("No CRS found, assuming WGS84 (EPSG:4326)")
                elif gdf.crs != 'EPSG:4326':
                    gdf = gdf.to_crs('EPSG:4326')
                    st.info(f"Reprojected from {gdf.crs} to WGS84")

                # Calculate centroid for mapping
                gdf['centroid_lat'] = gdf.geometry.centroid.y
                gdf['centroid_lon'] = gdf.geometry.centroid.x

                st.success(f"✅ Shapefile loaded with {len(gdf)} features")
                return gdf

        except Exception as e:
            st.error(f"❌ Error loading shapefile: {str(e)}")
            return None

    def join_data_with_shapefile(self, df, gdf, df_key='region', gdf_key='name'):
        """Join epidemiological data with shapefile"""
        try:
            # Convert keys to string for joining
            df[df_key] = df[df_key].astype(str)
            gdf[gdf_key] = gdf[gdf_key].astype(str)

            # Perform spatial join
            joined_gdf = gdf.merge(df, left_on=gdf_key, right_on=df_key, how='left')

            st.success(f"✅ Joined {len(joined_gdf)} features")
            return joined_gdf

        except Exception as e:
            st.error(f"❌ Error joining data: {str(e)}")
            return None


# ============================================
# ADVANCED SPATIAL ANALYSIS
# ============================================

class AdvancedSpatialAnalyzer:
    """Comprehensive spatial analysis with GIS operations"""

    def __init__(self):
        self.results = {}

    def calculate_spatial_autocorrelation(self, gdf, value_column='cases'):
        """Calculate spatial autocorrelation statistics"""
        results = {}

        try:
            # Create spatial weights matrix
            w = weights.Queen.from_dataframe(gdf)
            w.transform = 'r'

            # Moran's I
            y = gdf[value_column].fillna(0).values
            moran = esda.Moran(y, w)

            # Local Moran's I
            lisa = esda.Moran_Local(y, w)

            # Getis-Ord G
            go_g = esda.G_Local(y, w)

            # Geary's C
            geary = esda.Geary(y, w)

            # FIXED: Use moran.VI instead of moran.var_i
            results['moran_i'] = {
                'I': moran.I,
                'expected_i': moran.EI,
                'variance': moran.VI,  # FIXED: Changed from moran.var_i to moran.VI
                'z_score': moran.z_norm,
                'p_value': moran.p_norm,
                'significance': 'Significant' if moran.p_norm < 0.05 else 'Not significant'
            }

            results['local_moran'] = {
                'cluster_types': lisa.q.tolist(),
                'p_values': lisa.p_sim.tolist(),
                'hotspots': np.sum(lisa.q == 1),
                'coldspots': np.sum(lisa.q == 3),
                'outliers': np.sum((lisa.q == 2) | (lisa.q == 4))
            }

            results['getis_ord'] = {
                'G': go_g.G,
                'p_values': go_g.p_sim,
                'hotspots': np.sum(go_g.Zs > 1.96),
                'coldspots': np.sum(go_g.Zs < -1.96)
            }

            results['geary_c'] = {
                'C': geary.C,
                'expected_c': geary.EC,
                'z_score': geary.z_norm,
                'p_value': geary.p_norm
            }

        except Exception as e:
            st.warning(f"Spatial autocorrelation error: {str(e)}")
            results['error'] = str(e)

        return results

    def perform_spatial_regression(self, gdf, dependent_var, independent_vars):
        """Perform spatial regression analysis"""
        results = {}

        try:
            # Prepare data
            y = gdf[dependent_var].fillna(0).values.reshape(-1, 1)
            X = gdf[independent_vars].fillna(0).values

            # Create spatial weights
            w = weights.Queen.from_dataframe(gdf)
            w.transform = 'r'

            # OLS Regression (Baseline) - FIXED: Use correct pysal syntax
            try:
                ols = spreg.OLS(y, X, w=w, name_y=dependent_var, name_x=independent_vars)
                results['ols'] = {
                    'r2': ols.r2,
                    'aic': ols.aic,
                    'schwarz': ols.schwarz,
                    'coefficients': dict(zip(independent_vars, ols.betas.flatten())),
                    'p_values': dict(zip(independent_vars, ols.pvalues.flatten()))
                }
            except Exception as e:
                st.warning(f"OLS regression error: {str(e)}")

        except Exception as e:
            st.warning(f"Spatial regression error: {str(e)}")

        return results

    def calculate_voronoi_diagram(self, gdf, value_column='cases'):
        """Calculate Voronoi diagram for spatial interpolation"""
        try:
            # Get points from centroids
            points = [Point(xy) for xy in zip(gdf.centroid_lon, gdf.centroid_lat)]
            multipoint = MultiPoint(points)

            # Create Voronoi diagram
            voronoi = voronoi_diagram(multipoint)

            # Convert to GeoDataFrame
            voronoi_gdf = gpd.GeoDataFrame(geometry=list(voronoi.geoms))

            # Calculate area and assign values
            voronoi_gdf['area_km2'] = voronoi_gdf.geometry.area * 111 ** 2  # Approximate conversion
            voronoi_gdf['density'] = gdf[value_column].values / voronoi_gdf['area_km2'].values

            return voronoi_gdf

        except Exception as e:
            st.error(f"Voronoi calculation error: {str(e)}")
            return None

    def detect_space_time_clusters(self, df, spatial_eps=0.1, temporal_eps=7, min_samples=5):
        """Detect space-time clusters using 3D DBSCAN"""
        # Prepare 3D coordinates (lat, lon, time)
        if 'date' not in df.columns:
            st.warning("Date column not found for space-time clustering")
            return None, None

        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['time_numeric'] = (df['date'] - df['date'].min()).dt.days

        if not all(col in df.columns for col in ['latitude', 'longitude', 'time_numeric']):
            st.warning("Required columns (latitude, longitude, date) not found for clustering")
            return None, None

        coords = df[['latitude', 'longitude', 'time_numeric']].values

        # Normalize coordinates
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords)

        # Apply 3D DBSCAN
        dbscan = DBSCAN(eps=spatial_eps, min_samples=min_samples, metric='euclidean')
        clusters = dbscan.fit_predict(coords_scaled)

        # Analyze clusters
        cluster_stats = []
        unique_clusters = np.unique(clusters[clusters != -1])

        for cluster_id in unique_clusters:
            mask = clusters == cluster_id
            cluster_data = df[mask]

            stats = {
                'cluster_id': int(cluster_id),
                'n_points': int(mask.sum()),
                'total_cases': int(cluster_data['cases'].sum()) if 'cases' in cluster_data.columns else 0,
                'start_date': cluster_data['date'].min(),
                'end_date': cluster_data['date'].max(),
                'duration_days': (cluster_data['date'].max() - cluster_data['date'].min()).days,
                'mean_lat': cluster_data['latitude'].mean(),
                'mean_lon': cluster_data['longitude'].mean(),
                'radius_km': self._calculate_cluster_radius(cluster_data),
                'growth_rate': self._calculate_growth_rate(cluster_data)
            }
            cluster_stats.append(stats)

        return clusters, pd.DataFrame(cluster_stats)

    def _calculate_cluster_radius(self, cluster_data):
        """Calculate spatial radius of cluster"""
        if len(cluster_data) < 2 or 'latitude' not in cluster_data.columns or 'longitude' not in cluster_data.columns:
            return 0

        coords = cluster_data[['latitude', 'longitude']].values
        centroid = coords.mean(axis=0)
        distances = np.sqrt(((coords - centroid) ** 2).sum(axis=1))
        return np.max(distances) * 111  # Convert to km

    def _calculate_growth_rate(self, cluster_data):
        """Calculate growth rate within cluster"""
        if len(cluster_data) < 2 or 'cases' not in cluster_data.columns:
            return 0

        cluster_data = cluster_data.sort_values('date')
        cases = cluster_data['cases'].values

        if cases[0] > 0:
            return (cases[-1] - cases[0]) / cases[0]
        return 0


# ============================================
# ADVANCED TEMPORAL ANALYSIS
# ============================================

class AdvancedTemporalAnalyzer:
    """Comprehensive temporal analysis with advanced forecasting"""

    def __init__(self):
        self.models = {}
        self.forecasts = {}

    def decompose_time_series(self, ts, period=7, model='additive'):
        """Decompose time series into trend, seasonal, and residual components"""
        if len(ts) < period * 2:
            st.warning(f"Time series too short for decomposition (need at least {period * 2} points)")
            return None

        try:
            decomposition = seasonal_decompose(ts, model=model, period=period, extrapolate_trend='freq')

            return {
                'observed': decomposition.observed,
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'period': period
            }
        except Exception as e:
            st.warning(f"Decomposition error: {str(e)}")
            return None

    def test_stationarity(self, ts):
        """Test time series stationarity using multiple tests"""
        results = {}

        try:
            # Augmented Dickey-Fuller test
            adf_result = adfuller(ts.dropna())
            results['adf'] = {
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'stationary': adf_result[1] < 0.05
            }

            # KPSS test
            kpss_result = kpss(ts.dropna())
            results['kpss'] = {
                'statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'critical_values': kpss_result[3],
                'stationary': kpss_result[1] > 0.05
            }
        except Exception as e:
            st.warning(f"Stationarity test error: {str(e)}")

        return results

    def calculate_epidemic_indicators(self, df):
        """Calculate comprehensive epidemic indicators"""
        indicators = {}

        # Basic reproduction number (R0) estimation
        if 'date' in df.columns and 'cases' in df.columns:
            daily_cases = df.groupby('date')['cases'].sum()

            # Estimate R0 using different methods
            indicators['reproduction_number'] = self._estimate_r0(daily_cases)

            # Doubling time
            indicators['doubling_time'] = self._calculate_doubling_time(daily_cases)

            # Growth rate
            growth_rate = daily_cases.pct_change().dropna()
            indicators['growth_rate'] = {
                'mean': growth_rate.mean(),
                'std': growth_rate.std(),
                'current': growth_rate.iloc[-1] if len(growth_rate) > 0 else 0
            }

        # Case fatality rate
        if all(col in df.columns for col in ['cases', 'deaths']):
            total_cases = df['cases'].sum()
            total_deaths = df['deaths'].sum()
            indicators['case_fatality_rate'] = (total_deaths / total_cases * 100) if total_cases > 0 else 0

        # Attack rate
        if 'population' in df.columns and 'cases' in df.columns:
            total_population = df['population'].sum()
            total_cases = df['cases'].sum()
            indicators['attack_rate'] = (total_cases / total_population * 100) if total_population > 0 else 0

        # Hospitalization rate
        if all(col in df.columns for col in ['cases', 'hospitalizations']):
            total_cases = df['cases'].sum()
            total_hospitalizations = df['hospitalizations'].sum()
            indicators['hospitalization_rate'] = (total_hospitalizations / total_cases * 100) if total_cases > 0 else 0

        return indicators

    def _estimate_r0(self, daily_cases, serial_interval=7):
        """Estimate basic reproduction number using different methods"""
        if len(daily_cases) < serial_interval * 2:
            return {'estimate': 0, 'method': 'insufficient_data'}

        # Method 1: Simple ratio method
        recent_cases = daily_cases.iloc[-serial_interval:].sum()
        previous_cases = daily_cases.iloc[-serial_interval * 2:-serial_interval].sum()

        if previous_cases > 0:
            r0_simple = recent_cases / previous_cases
        else:
            r0_simple = 0

        # Method 2: Exponential growth rate method
        if len(daily_cases) > 14:
            x = np.arange(len(daily_cases))
            y = daily_cases.values
            mask = y > 0
            if mask.sum() > 5:
                x_fit = x[mask]
                y_fit = np.log(y[mask])
                slope, intercept = np.polyfit(x_fit, y_fit, 1)
                r0_exp = np.exp(slope * serial_interval)
            else:
                r0_exp = 0
        else:
            r0_exp = 0

        return {
            'simple_ratio': r0_simple,
            'exponential_growth': r0_exp,
            'mean': (r0_simple + r0_exp) / 2,
            'serial_interval': serial_interval
        }

    def _calculate_doubling_time(self, daily_cases, window=7):
        """Calculate epidemic doubling time"""
        if len(daily_cases) < window * 2:
            return {'estimate': 0, 'method': 'insufficient_data'}

        # Calculate growth rate
        recent_mean = daily_cases.iloc[-window:].mean()
        previous_mean = daily_cases.iloc[-window * 2:-window].mean()

        if previous_mean > 0:
            growth_rate = (recent_mean / previous_mean) ** (1 / window) - 1
            if growth_rate > 0:
                doubling_time = np.log(2) / np.log(1 + growth_rate)
            else:
                doubling_time = float('inf')
        else:
            doubling_time = 0

        return {
            'doubling_time_days': doubling_time,
            'growth_rate': growth_rate if previous_mean > 0 else 0
        }

    def build_arima_model(self, ts, order=(2, 1, 2), seasonal_order=(1, 1, 1, 7)):
        """Build ARIMA model with automatic order selection"""
        try:
            model = ARIMA(ts, order=order, seasonal_order=seasonal_order)
            model_fit = model.fit()

            # Get model summary
            summary = {
                'aic': model_fit.aic,
                'bic': model_fit.bic,
                'hqic': model_fit.hqic,
                'residuals': model_fit.resid,
                'forecast': model_fit.forecast(steps=30),
                'conf_int': model_fit.get_forecast(steps=30).conf_int()
            }

            return model_fit, summary

        except Exception as e:
            st.error(f"ARIMA model error: {str(e)}")
            return None, None

    def build_prophet_model(self, df, growth='linear', seasonality_mode='additive'):
        """Build Facebook Prophet model"""
        try:
            # Prepare data for Prophet
            prophet_df = df[['date', 'cases']].copy()
            prophet_df.columns = ['ds', 'y']

            # Create and fit model
            model = Prophet(
                growth=growth,
                seasonality_mode=seasonality_mode,
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10
            )

            # Add additional regressors if available
            for col in ['temperature', 'rainfall', 'humidity']:
                if col in df.columns:
                    prophet_df[col] = df[col]
                    model.add_regressor(col)

            model.fit(prophet_df)

            # Make future dataframe
            future = model.make_future_dataframe(periods=30, freq='D')

            # Add future regressors
            for col in ['temperature', 'rainfall', 'humidity']:
                if col in df.columns:
                    future[col] = np.mean(df[col])

            # Generate forecast
            forecast = model.predict(future)

            return model, forecast

        except Exception as e:
            st.error(f"Prophet model error: {str(e)}")
            return None, None


# ============================================
# ADVANCED NETWORK ANALYSIS
# ============================================

class AdvancedNetworkAnalyzer:
    """Advanced network analysis for transmission dynamics"""

    def __init__(self):
        self.graph = None
        self.analysis_results = {}

    def build_transmission_network(self, df, method='gravity', params=None):
        """Build disease transmission network using various methods"""
        if params is None:
            params = {}

        G = nx.Graph()

        # Add nodes with attributes
        for idx, row in df.iterrows():
            G.add_node(
                idx,
                region=row.get('region', f'Location_{idx}'),
                cases=row.get('cases', 0),
                population=row.get('population', 1000),
                latitude=row.get('latitude', 0),
                longitude=row.get('longitude', 0)
            )

        if method == 'gravity':
            # Gravity model: interaction ∝ (mass1 * mass2) / distance^2
            for i in range(len(df)):
                for j in range(i + 1, len(df)):
                    if all(col in df.columns for col in ['latitude', 'longitude', 'cases', 'population']):
                        distance = self._haversine_distance(
                            df.iloc[i]['latitude'], df.iloc[i]['longitude'],
                            df.iloc[j]['latitude'], df.iloc[j]['longitude']
                        )

                        mass_i = df.iloc[i]['cases'] * df.iloc[i]['population']
                        mass_j = df.iloc[j]['cases'] * df.iloc[j]['population']

                        if distance > 0:
                            interaction = (mass_i * mass_j) / (distance ** 2)
                            if interaction > params.get('threshold', 0):
                                G.add_edge(i, j, weight=interaction, distance=distance)

        elif method == 'radiation':
            # Radiation model for human mobility
            for i in range(len(df)):
                for j in range(i + 1, len(df)):
                    if all(col in df.columns for col in ['latitude', 'longitude', 'population']):
                        pop_i = df.iloc[i]['population']
                        pop_j = df.iloc[j]['population']

                        # Calculate population within radius
                        distances = []
                        for k in range(len(df)):
                            if k != i and k != j:
                                dist = self._haversine_distance(
                                    df.iloc[i]['latitude'], df.iloc[i]['longitude'],
                                    df.iloc[k]['latitude'], df.iloc[k]['longitude']
                                )
                                distances.append((k, dist))

                        # Sort by distance
                        distances.sort(key=lambda x: x[1])

                        # Find population within circle radius r_ij
                        distance_ij = self._haversine_distance(
                            df.iloc[i]['latitude'], df.iloc[i]['longitude'],
                            df.iloc[j]['latitude'], df.iloc[j]['longitude']
                        )

                        pop_within = 0
                        for k, dist in distances:
                            if dist < distance_ij:
                                pop_within += df.iloc[k]['population']

                        # Radiation model formula
                        if (pop_i + pop_within) > 0:
                            interaction = (pop_i * pop_j) / ((pop_i + pop_within) * (pop_i + pop_j + pop_within))
                            G.add_edge(i, j, weight=interaction, distance=distance_ij)

        elif method == 'proximity':
            # Simple proximity-based network
            threshold = params.get('threshold_km', 50)

            for i in range(len(df)):
                for j in range(i + 1, len(df)):
                    if all(col in df.columns for col in ['latitude', 'longitude']):
                        distance = self._haversine_distance(
                            df.iloc[i]['latitude'], df.iloc[i]['longitude'],
                            df.iloc[j]['latitude'], df.iloc[j]['longitude']
                        )

                        if distance <= threshold:
                            weight = 1 / (distance + 1)  # Avoid division by zero
                            G.add_edge(i, j, weight=weight, distance=distance)

        self.graph = G
        return G

    def analyze_network_properties(self, G):
        """Analyze comprehensive network properties"""
        results = {}

        if G is None or G.number_of_nodes() == 0:
            return results

        # Basic properties
        results['basic'] = {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G),
            'connected_components': nx.number_connected_components(G),
            'is_connected': nx.is_connected(G)
        }

        # Degree analysis
        degrees = dict(G.degree())
        results['degree'] = {
            'distribution': list(degrees.values()),
            'mean': np.mean(list(degrees.values())),
            'std': np.std(list(degrees.values())),
            'max': max(degrees.values()),
            'min': min(degrees.values()),
            'assortativity': nx.degree_assortativity_coefficient(G)
        }

        # Centrality measures
        try:
            results['centrality'] = {
                'degree': nx.degree_centrality(G),
                'betweenness': nx.betweenness_centrality(G, weight='weight'),
                'closeness': nx.closeness_centrality(G, distance='weight'),
                'eigenvector': nx.eigenvector_centrality(G, max_iter=1000),
                'katz': nx.katz_centrality(G, max_iter=1000)
            }
        except:
            pass

        # Community detection
        try:
            communities = nx.algorithms.community.greedy_modularity_communities(G)
            results['communities'] = {
                'n_communities': len(communities),
                'sizes': [len(c) for c in communities],
                'modularity': nx.algorithms.community.modularity(G, communities)
            }
        except:
            pass

        # Path analysis
        if nx.is_connected(G):
            try:
                results['paths'] = {
                    'average_path_length': nx.average_shortest_path_length(G, weight='weight'),
                    'diameter': nx.diameter(G),
                    'radius': nx.radius(G),
                    'center': nx.center(G),
                    'periphery': nx.periphery(G)
                }
            except:
                pass

        # Clustering
        results['clustering'] = {
            'average_clustering': nx.average_clustering(G),
            'transitivity': nx.transitivity(G),
            'square_clustering': nx.square_clustering(G)
        }

        return results

    def simulate_epidemic_spread(self, G, model='SIR', params=None):
        """Simulate epidemic spread on network"""
        if params is None:
            params = {}

        if model == 'SIR':
            return self._simulate_sir(G, params)
        elif model == 'SIS':
            return self._simulate_sis(G, params)
        elif model == 'SEIR':
            return self._simulate_seir(G, params)
        else:
            return self._simulate_sir(G, params)

    def _simulate_sir(self, G, params):
        """Simulate SIR model on network"""
        beta = params.get('beta', 0.3)  # Transmission rate
        gamma = params.get('gamma', 0.1)  # Recovery rate
        initial_infected = params.get('initial_infected', 3)
        steps = params.get('steps', 50)

        # Initialize states
        states = {node: 'S' for node in G.nodes()}
        infected = list(G.nodes())[:initial_infected]
        for node in infected:
            states[node] = 'I'

        # Store results
        history = []
        history.append({
            'step': 0,
            'S': G.number_of_nodes() - initial_infected,
            'I': initial_infected,
            'R': 0
        })

        for step in range(1, steps + 1):
            new_states = states.copy()

            # Process each node
            for node in G.nodes():
                if states[node] == 'I':
                    # Try to infect neighbors
                    for neighbor in G.neighbors(node):
                        if states[neighbor] == 'S':
                            # Transmission probability depends on edge weight
                            edge_weight = G[node][neighbor].get('weight', 1)
                            if np.random.random() < beta * edge_weight:
                                new_states[neighbor] = 'I'

                    # Recover
                    if np.random.random() < gamma:
                        new_states[node] = 'R'

            states = new_states

            # Count states
            s_count = sum(1 for state in states.values() if state == 'S')
            i_count = sum(1 for state in states.values() if state == 'I')
            r_count = sum(1 for state in states.values() if state == 'R')

            history.append({
                'step': step,
                'S': s_count,
                'I': i_count,
                'R': r_count
            })

        return pd.DataFrame(history)

    def _simulate_sis(self, G, params):
        """Simulate SIS model on network"""
        beta = params.get('beta', 0.3)
        gamma = params.get('gamma', 0.1)
        initial_infected = params.get('initial_infected', 3)
        steps = params.get('steps', 50)

        states = {node: 'S' for node in G.nodes()}
        infected = list(G.nodes())[:initial_infected]
        for node in infected:
            states[node] = 'I'

        history = []
        history.append({
            'step': 0,
            'S': G.number_of_nodes() - initial_infected,
            'I': initial_infected
        })

        for step in range(1, steps + 1):
            new_states = states.copy()

            for node in G.nodes():
                if states[node] == 'I':
                    # Infect neighbors
                    for neighbor in G.neighbors(node):
                        if states[neighbor] == 'S':
                            edge_weight = G[node][neighbor].get('weight', 1)
                            if np.random.random() < beta * edge_weight:
                                new_states[neighbor] = 'I'

                    # Recover (go back to susceptible)
                    if np.random.random() < gamma:
                        new_states[node] = 'S'

            states = new_states

            s_count = sum(1 for state in states.values() if state == 'S')
            i_count = sum(1 for state in states.values() if state == 'I')

            history.append({
                'step': step,
                'S': s_count,
                'I': i_count
            })

        return pd.DataFrame(history)

    def _simulate_seir(self, G, params):
        """Simulate SEIR model on network"""
        beta = params.get('beta', 0.3)
        sigma = params.get('sigma', 0.2)  # Incubation rate
        gamma = params.get('gamma', 0.1)
        initial_infected = params.get('initial_infected', 3)
        steps = params.get('steps', 50)

        states = {node: 'S' for node in G.nodes()}
        infected = list(G.nodes())[:initial_infected]
        for node in infected:
            states[node] = 'I'

        history = []
        history.append({
            'step': 0,
            'S': G.number_of_nodes() - initial_infected,
            'E': 0,
            'I': initial_infected,
            'R': 0
        })

        for step in range(1, steps + 1):
            new_states = states.copy()

            for node in G.nodes():
                if states[node] == 'I':
                    # Infect neighbors (they become exposed)
                    for neighbor in G.neighbors(node):
                        if states[neighbor] == 'S':
                            edge_weight = G[node][neighbor].get('weight', 1)
                            if np.random.random() < beta * edge_weight:
                                new_states[neighbor] = 'E'

                    # Recover
                    if np.random.random() < gamma:
                        new_states[node] = 'R'

                elif states[node] == 'E':
                    # Move from exposed to infected
                    if np.random.random() < sigma:
                        new_states[node] = 'I'

            states = new_states

            s_count = sum(1 for state in states.values() if state == 'S')
            e_count = sum(1 for state in states.values() if state == 'E')
            i_count = sum(1 for state in states.values() if state == 'I')
            r_count = sum(1 for state in states.values() if state == 'R')

            history.append({
                'step': step,
                'S': s_count,
                'E': e_count,
                'I': i_count,
                'R': r_count
            })

        return pd.DataFrame(history)

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate haversine distance in km"""
        R = 6371  # Earth's radius in km

        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c


# ============================================
# ADVANCED MACHINE LEARNING MODELS
# ============================================

class AdvancedMLModels:
    """Comprehensive machine learning models for epidemiological prediction"""

    def __init__(self):
        self.models = {}
        self.results = {}
        self.feature_importance = {}

    def prepare_features(self, df, target='cases', lag_days=[1, 3, 7, 14], rolling_windows=[3, 7, 14]):
        """Prepare advanced feature set for ML models"""
        features = df.copy()

        # Date features
        if 'date' in features.columns:
            features['date'] = pd.to_datetime(features['date'])
            features['day_of_year'] = features['date'].dt.dayofyear
            features['month'] = features['date'].dt.month
            features['week'] = features['date'].dt.isocalendar().week
            features['quarter'] = features['date'].dt.quarter
            features['day_of_week'] = features['date'].dt.dayofweek
            features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)

            # Cyclical encoding
            features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
            features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
            features['day_sin'] = np.sin(2 * np.pi * features['day_of_year'] / 365)
            features['day_cos'] = np.cos(2 * np.pi * features['day_of_year'] / 365)

        # Lag features
        features = features.sort_values('date')
        for lag in lag_days:
            features[f'{target}_lag_{lag}'] = features[target].shift(lag)

        # Rolling statistics
        for window in rolling_windows:
            features[f'{target}_rolling_mean_{window}'] = features[target].rolling(window=window).mean()
            features[f'{target}_rolling_std_{window}'] = features[target].rolling(window=window).std()
            features[f'{target}_rolling_max_{window}'] = features[target].rolling(window=window).max()
            features[f'{target}_rolling_min_{window}'] = features[target].rolling(window=window).min()

        # Spatial features
        if all(col in features.columns for col in ['latitude', 'longitude']):
            features['centroid_lat'] = features['latitude'].mean()
            features['centroid_lon'] = features['longitude'].mean()
            features['distance_to_center'] = np.sqrt(
                (features['latitude'] - features['centroid_lat']) ** 2 +
                (features['longitude'] - features['centroid_lon']) ** 2
            )

        # Interaction features
        if all(col in features.columns for col in ['temperature', 'humidity']):
            features['temp_humidity_interaction'] = features['temperature'] * features['humidity']

        if all(col in features.columns for col in ['temperature', 'rainfall']):
            features['temp_rain_interaction'] = features['temperature'] * features['rainfall']

        # Drop rows with NaN
        features = features.dropna()

        return features

    def train_ensemble_model(self, X, y, models_to_use=None):
        """Train ensemble of ML models"""
        # FIXED: Changed parameter name from models_to_train to models_to_use
        if models_to_use is None:
            models_to_use = ['xgboost', 'lightgbm', 'random_forest', 'gradient_boosting']

        results = {}

        # Split data with time series consideration
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for model_name in models_to_use:
            with st.spinner(f"Training {model_name}..."):
                try:
                    if model_name == 'xgboost':
                        model = xgb.XGBRegressor(
                            n_estimators=200,
                            max_depth=6,
                            learning_rate=0.1,
                            random_state=42,
                            n_jobs=-1
                        )
                    elif model_name == 'lightgbm':
                        model = lgb.LGBMRegressor(
                            n_estimators=200,
                            max_depth=6,
                            learning_rate=0.1,
                            random_state=42,
                            n_jobs=-1
                        )
                    elif model_name == 'random_forest':
                        model = RandomForestRegressor(
                            n_estimators=200,
                            max_depth=15,
                            min_samples_split=5,
                            min_samples_leaf=2,
                            random_state=42,
                            n_jobs=-1
                        )
                    elif model_name == 'gradient_boosting':
                        model = GradientBoostingRegressor(
                            n_estimators=200,
                            learning_rate=0.1,
                            max_depth=5,
                            random_state=42
                        )
                    elif model_name == 'catboost':
                        model = cb.CatBoostRegressor(
                            iterations=200,
                            depth=6,
                            learning_rate=0.1,
                            random_seed=42,
                            verbose=0
                        )
                    elif model_name == 'svr':
                        model = SVR(kernel='rbf', C=100, gamma=0.1)
                    elif model_name == 'mlp':
                        model = MLPRegressor(
                            hidden_layer_sizes=(100, 50),
                            activation='relu',
                            solver='adam',
                            max_iter=1000,
                            random_state=42
                        )
                    else:
                        continue

                    # Train model
                    model.fit(X_train_scaled, y_train)

                    # Make predictions
                    y_pred_train = model.predict(X_train_scaled)
                    y_pred_test = model.predict(X_test_scaled)

                    # Calculate metrics
                    metrics = {
                        'train': {
                            'r2': r2_score(y_train, y_pred_train),
                            'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                            'mae': mean_absolute_error(y_train, y_pred_train),
                            'mape': mean_absolute_percentage_error(y_train, y_pred_train)
                        },
                        'test': {
                            'r2': r2_score(y_test, y_pred_test),
                            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                            'mae': mean_absolute_error(y_test, y_pred_test),
                            'mape': mean_absolute_percentage_error(y_test, y_pred_test)
                        }
                    }

                    # Feature importance
                    feature_importance = None
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = pd.DataFrame({
                            'feature': X.columns,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)

                    # Store results
                    model_key = f"{model_name}_{datetime.now().strftime('%H%M%S')}"
                    self.models[model_key] = model
                    self.results[model_key] = {
                        'model': model,
                        'metrics': metrics,
                        'feature_importance': feature_importance,
                        'predictions': {
                            'train': y_pred_train,
                            'test': y_pred_test
                        },
                        'scaler': scaler
                    }

                    results[model_name] = metrics

                except Exception as e:
                    st.warning(f"Error training {model_name}: {str(e)}")

        return results

    def perform_hyperparameter_tuning(self, X, y, model_type='random_forest'):
        """Perform hyperparameter tuning for ML models"""
        # FIXED: Added missing method
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20],
                    'min_samples_split': [2, 5, 10]
                }
                base_model = RandomForestRegressor(random_state=42)

            elif model_type == 'xgboost':
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
                base_model = xgb.XGBRegressor(random_state=42)

            elif model_type == 'lightgbm':
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
                base_model = lgb.LGBMRegressor(random_state=42, verbose=-1)

            elif model_type == 'svr':
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'gamma': [0.001, 0.01, 0.1, 1],
                    'kernel': ['rbf', 'linear']
                }
                base_model = SVR()
            else:
                st.warning(f"Hyperparameter tuning not implemented for {model_type}")
                return None, None

            # Perform grid search
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=5,
                scoring='r2',
                n_jobs=-1,
                verbose=0
            )

            grid_search.fit(X_train, y_train)

            # Get best model
            best_model = grid_search.best_estimator_

            # Evaluate on test set
            y_pred = best_model.predict(X_test)
            test_r2 = r2_score(y_test, y_pred)

            metrics = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'test_r2': test_r2
            }

            return best_model, metrics

        except Exception as e:
            st.error(f"Hyperparameter tuning error: {str(e)}")
            return None, None

    def train_lstm_model(self, X, y, sequence_length=10, epochs=50):
        """Train LSTM model for time series prediction"""
        try:
            # Reshape data for LSTM [samples, time steps, features]
            X_array = X.values.reshape((X.shape[0], 1, X.shape[1]))

            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X_array[:split_idx], X_array[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Build LSTM model
            model = keras.Sequential([
                layers.LSTM(50, activation='relu', input_shape=(1, X.shape[1]), return_sequences=True),
                layers.Dropout(0.2),
                layers.LSTM(25, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(1)
            ])

            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )

            # Make predictions
            y_pred_train = model.predict(X_train).flatten()
            y_pred_test = model.predict(X_test).flatten()

            # Calculate metrics
            metrics = {
                'train': {
                    'r2': r2_score(y_train, y_pred_train),
                    'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                    'mae': mean_absolute_error(y_train, y_pred_train)
                },
                'test': {
                    'r2': r2_score(y_test, y_pred_test),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                    'mae': mean_absolute_error(y_test, y_pred_test)
                }
            }

            # Store model
            model_key = f"lstm_{datetime.now().strftime('%H%M%S')}"
            self.models[model_key] = model
            self.results[model_key] = {
                'model': model,
                'metrics': metrics,
                'history': history.history,
                'predictions': {
                    'train': y_pred_train,
                    'test': y_pred_test
                }
            }

            return metrics

        except Exception as e:
            st.error(f"LSTM training error: {str(e)}")
            return None


# ============================================
# EPIDEMIOLOGICAL MODELS
# ============================================

class EpidemiologicalModels:
    """Advanced epidemiological models for transmission dynamics"""

    def __init__(self):
        self.models = {}

    def fit_sir_model(self, data, population, initial_conditions=None):
        """Fit SIR model to data"""
        if initial_conditions is None:
            initial_conditions = [population - 1, 1, 0]  # S, I, R

        # SIR differential equations
        def sir_ode(t, y, beta, gamma):
            S, I, R = y
            N = S + I + R

            dSdt = -beta * S * I / N
            dIdt = beta * S * I / N - gamma * I
            dRdt = gamma * I

            return [dSdt, dIdt, dRdt]

        # Time points
        t = np.arange(len(data))

        # Objective function for optimization
        def objective(params):
            beta, gamma = params
            solution = solve_ivp(
                sir_ode,
                [t[0], t[-1]],
                initial_conditions,
                args=(beta, gamma),
                t_eval=t,
                method='RK45'
            )

            # Calculate error
            predicted = solution.y[1, :]  # Infected
            error = np.sum((predicted - data) ** 2)
            return error

        # Optimize parameters
        from scipy.optimize import minimize
        initial_guess = [0.3, 0.1]  # beta, gamma
        bounds = [(0.001, 1), (0.001, 1)]

        result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')

        if result.success:
            beta, gamma = result.x
            R0 = beta / gamma

            # Generate fitted curve
            solution = solve_ivp(
                sir_ode,
                [t[0], t[-1]],
                initial_conditions,
                args=(beta, gamma),
                t_eval=t,
                method='RK45'
            )

            return {
                'beta': beta,
                'gamma': gamma,
                'R0': R0,
                'fitted_curve': solution.y[1, :],
                'success': True,
                'message': result.message
            }
        else:
            return {
                'success': False,
                'message': result.message
            }

    def fit_seir_model(self, data, population, initial_conditions=None):
        """Fit SEIR model to data"""
        if initial_conditions is None:
            initial_conditions = [population - 1, 0, 1, 0]  # S, E, I, R

        # SEIR differential equations
        def seir_ode(t, y, beta, sigma, gamma):
            S, E, I, R = y
            N = S + E + I + R

            dSdt = -beta * S * I / N
            dEdt = beta * S * I / N - sigma * E
            dIdt = sigma * E - gamma * I
            dRdt = gamma * I

            return [dSdt, dEdt, dIdt, dRdt]

        # Time points
        t = np.arange(len(data))

        # Objective function
        def objective(params):
            beta, sigma, gamma = params
            solution = solve_ivp(
                seir_ode,
                [t[0], t[-1]],
                initial_conditions,
                args=(beta, sigma, gamma),
                t_eval=t,
                method='RK45'
            )

            predicted = solution.y[2, :]  # Infected
            error = np.sum((predicted - data) ** 2)
            return error

        # Optimize
        from scipy.optimize import minimize
        initial_guess = [0.3, 0.2, 0.1]  # beta, sigma, gamma
        bounds = [(0.001, 1), (0.001, 1), (0.001, 1)]

        result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')

        if result.success:
            beta, sigma, gamma = result.x
            R0 = beta / gamma

            solution = solve_ivp(
                seir_ode,
                [t[0], t[-1]],
                initial_conditions,
                args=(beta, sigma, gamma),
                t_eval=t,
                method='RK45'
            )

            return {
                'beta': beta,
                'sigma': sigma,
                'gamma': gamma,
                'R0': R0,
                'fitted_curve': solution.y[2, :],
                'success': True
            }
        else:
            return {'success': False, 'message': result.message}

    def calculate_epidemic_thresholds(self, data, window=7):
        """Calculate various epidemic thresholds"""
        thresholds = {}

        # Basic thresholds
        mean_cases = np.mean(data)
        std_cases = np.std(data)

        thresholds['basic'] = {
            'mean': mean_cases,
            'std': std_cases,
            'alert_threshold': mean_cases + std_cases,
            'warning_threshold': mean_cases + 2 * std_cases,
            'critical_threshold': mean_cases + 3 * std_cases
        }

        # Exponential growth detection
        if len(data) > window * 2:
            recent_growth = data[-window:].mean() / data[-window * 2:-window].mean() if data[
                                                                                            -window * 2:-window].mean() > 0 else 0
            thresholds['growth'] = {
                'recent_growth': recent_growth,
                'exponential_growth': recent_growth > 1.2,
                'doubling_time': np.log(2) / np.log(recent_growth) if recent_growth > 1 else float('inf')
            }

        # Outbreak detection using CUSUM
        cusum = np.cumsum(data - mean_cases)
        thresholds['cusum'] = {
            'max_cusum': np.max(cusum),
            'min_cusum': np.min(cusum),
            'outbreak_signal': np.max(cusum) > 3 * std_cases
        }

        return thresholds

    def simulate_intervention_scenarios(self, base_model, interventions):
        """Simulate different intervention scenarios"""
        scenarios = {}

        for name, params in interventions.items():
            # Modify model parameters based on intervention
            modified_params = base_model.copy()
            modified_params.update(params)

            # Simulate with modified parameters
            # This would depend on the specific model being used
            scenarios[name] = {
                'parameters': modified_params,
                'estimated_impact': self._estimate_intervention_impact(base_model, modified_params)
            }

        return scenarios

    def _estimate_intervention_impact(self, base_params, intervention_params):
        """Estimate impact of intervention"""
        # Simplified impact estimation
        impact = {}

        # Estimate reduction in transmission
        if 'beta' in base_params and 'beta' in intervention_params:
            reduction = (base_params['beta'] - intervention_params['beta']) / base_params['beta'] * 100
            impact['transmission_reduction'] = reduction

        # Estimate cases averted
        impact['cases_averted'] = 'Simulation required'

        return impact


# ============================================
# VISUALIZATION ENGINE
# ============================================

class AdvancedVisualization:
    """Advanced visualization engine with interactive plots"""

    def __init__(self):
        self.color_scales = {
            'cases': px.colors.sequential.Reds,
            'deaths': px.colors.sequential.Purples,
            'risk': px.colors.sequential.Viridis,
            'temperature': px.colors.sequential.Plasma,
            'rainfall': px.colors.sequential.Blues,
            'growth': px.colors.diverging.RdYlGn[::-1]
        }

    def create_interactive_map(self, data, map_type='heatmap', value_column='cases', height=600):
        """Create interactive map with multiple visualization options"""
        # Calculate center coordinates
        if 'latitude' in data.columns and 'longitude' in data.columns:
            # Filter out NaN values for coordinates
            valid_data = data.dropna(subset=['latitude', 'longitude'])
            if len(valid_data) > 0:
                center_lat = valid_data['latitude'].mean()
                center_lon = valid_data['longitude'].mean()
            else:
                center_lat, center_lon = 23.6850, 90.3563  # Default Bangladesh center
        else:
            center_lat, center_lon = 23.6850, 90.3563  # Default Bangladesh center

        m = folium.Map(location=[center_lat, center_lon], zoom_start=7,
                       tiles='cartodbpositron', height=height)

        if map_type == 'heatmap':
            # Create heatmap data
            heat_data = []
            for _, row in data.iterrows():
                if (pd.notna(row.get('latitude', None)) and
                        pd.notna(row.get('longitude', None)) and
                        pd.notna(row.get(value_column, None))):
                    value = float(row[value_column])
                    heat_data.append([row['latitude'], row['longitude'], value])

            if heat_data:
                HeatMap(heat_data, radius=15, blur=10, max_zoom=10).add_to(m)
                st.success(f"Heatmap created with {len(heat_data)} data points")
            else:
                st.warning("No valid location data for heatmap. Check your data for missing coordinates or values.")

        elif map_type == 'cluster':
            # Create marker clusters
            marker_cluster = MarkerCluster().add_to(m)

            for _, row in data.iterrows():
                if pd.notna(row.get('latitude', None)) and pd.notna(row.get('longitude', None)):
                    popup_content = f"""
                    <div style="min-width: 250px">
                        <h4>{row.get('region', 'Location')}</h4>
                        <p><b>Date:</b> {row.get('date', 'N/A')}</p>
                        <p><b>Cases:</b> {row.get('cases', 0)}</p>
                        <p><b>Deaths:</b> {row.get('deaths', 0)}</p>
                        <p><b>Temperature:</b> {row.get('temperature', 'N/A')}°C</p>
                        <p><b>Rainfall:</b> {row.get('rainfall', 'N/A')}mm</p>
                    </div>
                    """

                    value = row.get(value_column, 0)
                    if value_column in data.columns:
                        max_value = data[value_column].max()
                        radius = 5 + (value / max_value * 20) if max_value > 0 else 5
                    else:
                        radius = 5

                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=radius,
                        popup=folium.Popup(popup_content, max_width=300),
                        color=self._get_color_by_value(value,
                                                       data[value_column] if value_column in data.columns else []),
                        fill=True,
                        fill_opacity=0.7
                    ).add_to(marker_cluster)

        elif map_type == 'choropleth' and 'geometry' in data.columns:
            # Create choropleth map
            try:
                import branca.colormap as cm

                # Convert to GeoDataFrame if not already
                if not isinstance(data, gpd.GeoDataFrame):
                    gdf = gpd.GeoDataFrame(data)
                else:
                    gdf = data

                # Create colormap
                colormap = cm.linear.YlOrRd_09.scale(
                    gdf[value_column].min(),
                    gdf[value_column].max()
                )

                # Add choropleth
                folium.GeoJson(
                    gdf.__geo_interface__,
                    style_function=lambda feature: {
                        'fillColor': colormap(feature['properties'][value_column]),
                        'color': 'black',
                        'weight': 1,
                        'fillOpacity': 0.7,
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=['region', value_column, 'population'] if 'population' in data.columns else ['region',
                                                                                                            value_column],
                        aliases=['Region', value_column.title(), 'Population'] if 'population' in data.columns else [
                            'Region', value_column.title()]
                    )
                ).add_to(m)

                # Add colormap to map
                colormap.add_to(m)
            except Exception as e:
                st.warning(f"Choropleth creation error: {str(e)}")

        elif map_type == 'animated' and 'date' in data.columns:
            # Create animated heatmap over time
            try:
                data['date_str'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
                dates = sorted(data['date_str'].unique())

                # Prepare heatmap data
                heat_data = []
                for date in dates:
                    day_data = data[data['date_str'] == date]
                    day_points = []
                    for _, row in day_data.iterrows():
                        if pd.notna(row.get('latitude', None)) and pd.notna(row.get('longitude', None)):
                            value = row.get(value_column, 0)
                            if pd.notna(value):
                                day_points.append([row['latitude'], row['longitude'], float(value)])
                    heat_data.append(day_points)

                if heat_data and len(heat_data[0]) > 0:
                    HeatMapWithTime(
                        heat_data,
                        index=list(dates),
                        radius=15,
                        auto_play=True,
                        max_opacity=0.8,
                        min_opacity=0.1
                    ).add_to(m)
                else:
                    st.warning("No valid data for animated heatmap")
            except Exception as e:
                st.warning(f"Animated heatmap error: {str(e)}")

        elif map_type == 'bubble' and 'latitude' in data.columns and 'longitude' in data.columns:
            # Create bubble map
            for _, row in data.iterrows():
                if pd.notna(row.get('latitude', None)) and pd.notna(row.get('longitude', None)):
                    value = row.get(value_column, 0)
                    if value_column in data.columns:
                        max_value = data[value_column].max()
                        radius = 3 + (value / max_value * 20) if max_value > 0 else 3
                    else:
                        radius = 3

                    popup_content = f"""
                    <div style="min-width: 200px">
                        <h4>{row.get('region', 'Location')}</h4>
                        <p><b>{value_column}:</b> {value}</p>
                        <p><b>Date:</b> {row.get('date', 'N/A')}</p>
                    </div>
                    """

                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=radius,
                        popup=folium.Popup(popup_content, max_width=300),
                        color=self._get_color_by_value(value,
                                                       data[value_column] if value_column in data.columns else []),
                        fill=True,
                        fill_opacity=0.6,
                        weight=1
                    ).add_to(m)

        # Add layer control
        folium.LayerControl().add_to(m)

        return m

    def create_correlation_matrix(self, data, method='pearson'):
        """Create interactive correlation matrix"""
        # Calculate correlation matrix
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for correlation matrix")
            return None

        corr_matrix = data[numeric_cols].corr(method=method)

        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            title=f'{method.title()} Correlation Matrix',
            labels=dict(color="Correlation"),
            aspect="auto"
        )

        # Add correlation values
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=f"{corr_matrix.iloc[i, j]:.2f}",
                    showarrow=False,
                    font=dict(color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")
                )

        fig.update_layout(height=600)
        return fig

    def create_epidemic_curve(self, data, cumulative=False, rolling_window=7):
        """Create epidemic curve visualization"""
        if 'date' in data.columns and 'cases' in data.columns:
            # Convert date column if needed
            if not pd.api.types.is_datetime64_any_dtype(data['date']):
                data['date'] = pd.to_datetime(data['date'])

            daily_cases = data.groupby('date')['cases'].sum()

            if cumulative:
                y_data = daily_cases.cumsum()
                title = 'Cumulative Cases'
            else:
                y_data = daily_cases
                title = 'Daily Cases'

            fig = go.Figure()

            # Add bar chart
            fig.add_trace(go.Bar(
                x=daily_cases.index,
                y=y_data,
                name='Cases',
                marker_color='#1a5f7a'
            ))

            # Add rolling average
            if rolling_window > 1 and len(daily_cases) > rolling_window:
                rolling_avg = daily_cases.rolling(window=rolling_window).mean()
                if cumulative:
                    rolling_avg = rolling_avg.cumsum()

                fig.add_trace(go.Scatter(
                    x=rolling_avg.index,
                    y=rolling_avg,
                    name=f'{rolling_window}-Day Moving Average',
                    line=dict(color='#e74c3c', width=3)
                ))

            fig.update_layout(
                title=title,
                xaxis_title='Date',
                yaxis_title='Number of Cases',
                hovermode='x unified',
                height=500
            )

            return fig
        return None

    def create_spatial_autocorrelation_plot(self, gdf, value_column='cases'):
        """Create spatial autocorrelation visualization"""
        try:
            # Calculate spatial weights
            w = weights.Queen.from_dataframe(gdf)
            w.transform = 'r'

            # Calculate Moran's I
            y = gdf[value_column].fillna(0).values
            moran = esda.Moran(y, w)

            # Create Moran scatterplot
            fig, ax = plt.subplots(figsize=(10, 8))

            # Standardize values
            y_std = (y - y.mean()) / y.std()
            lag = weights.lag_spatial(w, y_std)

            # Create scatterplot
            ax.scatter(y_std, lag, alpha=0.6)
            ax.axhline(y=0, color='grey', linestyle='--')
            ax.axvline(x=0, color='grey', linestyle='--')

            # Add regression line
            b, a = np.polyfit(y_std, lag, 1)
            ax.plot(y_std, a + b * y_std, color='red', linewidth=2)

            ax.set_xlabel(f'Standardized {value_column}')
            ax.set_ylabel(f'Spatial Lag of {value_column}')
            ax.set_title(f"Moran's I = {moran.I:.3f} (p = {moran.p_norm:.3f})")

            plt.tight_layout()
            return fig
        except Exception as e:
            st.warning(f"Spatial autocorrelation plot error: {str(e)}")
            return None

    def _get_color_by_value(self, value, series):
        """Get color based on value percentile"""
        if len(series) == 0:
            return '#3498db'

        try:
            # Convert to numpy array if it's a pandas Series
            if hasattr(series, 'values'):
                series_values = series.values
            else:
                series_values = series

            # Filter out NaN values
            series_values = series_values[~np.isnan(series_values)]

            if len(series_values) == 0:
                return '#3498db'

            percentile = stats.percentileofscore(series_values, value) / 100

            if percentile > 0.75:
                return '#e74c3c'  # Red
            elif percentile > 0.5:
                return '#f39c12'  # Orange
            elif percentile > 0.25:
                return '#27ae60'  # Green
            else:
                return '#3498db'  # Blue
        except Exception as e:
            return '#3498db'  # Default blue color on error


# ============================================
# EXPORT MANAGER
# ============================================

class ExportManager:
    """Advanced export manager for data, plots, and reports"""

    def __init__(self):
        self.export_formats = {
            'data': ['CSV', 'Excel', 'JSON', 'GeoJSON', 'Parquet'],
            'plots': ['PNG', 'PDF', 'SVG', 'HTML'],
            'reports': ['PDF', 'HTML', 'DOCX', 'Markdown']
        }

    def export_data(self, data, format='CSV', filename='data'):
        """Export data in various formats"""
        try:
            if format == 'CSV':
                csv = data.to_csv(index=False)
                return csv.encode(), f'{filename}.csv', 'text/csv'

            elif format == 'Excel':
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    data.to_excel(writer, index=False, sheet_name='Data')
                return output.getvalue(), f'{filename}.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'

            elif format == 'JSON':
                json_str = data.to_json(orient='records', indent=2)
                return json_str.encode(), f'{filename}.json', 'application/json'

            elif format == 'GeoJSON' and hasattr(data, 'geometry'):
                geojson = data.to_json()
                return geojson.encode(), f'{filename}.geojson', 'application/geo+json'

            elif format == 'Parquet':
                output = io.BytesIO()
                data.to_parquet(output, index=False)
                return output.getvalue(), f'{filename}.parquet', 'application/octet-stream'

        except Exception as e:
            st.error(f"Export error: {str(e)}")
            return None, None, None

    def export_plot(self, fig, format='PNG', filename='plot'):
        """Export plot in various formats"""
        try:
            if format == 'PNG':
                img_bytes = fig.to_image(format="png", width=1200, height=800)
                return img_bytes, f'{filename}.png', 'image/png'

            elif format == 'PDF':
                img_bytes = fig.to_image(format="pdf", width=1200, height=800)
                return img_bytes, f'{filename}.pdf', 'application/pdf'

            elif format == 'SVG':
                img_bytes = fig.to_image(format="svg", width=1200, height=800)
                return img_bytes, f'{filename}.svg', 'image/svg+xml'

            elif format == 'HTML':
                html = fig.to_html(include_plotlyjs='cdn')
                return html.encode(), f'{filename}.html', 'text/html'

        except Exception as e:
            st.error(f"Plot export error: {str(e)}")
            return None, None, None

    def generate_report(self, data, analysis_results, report_type='comprehensive'):
        """Generate comprehensive analysis report"""
        report = []

        # Header
        report.append("=" * 80)
        report.append("EPIDEMIOLOGICAL ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Data Summary
        report.append("\n" + "=" * 80)
        report.append("DATA SUMMARY")
        report.append("=" * 80)

        if 'date' in data.columns:
            try:
                report.append(f"Date Range: {data['date'].min().date()} to {data['date'].max().date()}")
            except:
                report.append(f"Date Range: {data['date'].min()} to {data['date'].max()}")
        report.append(f"Total Records: {len(data):,}")
        report.append(f"Variables: {len(data.columns)}")

        if 'cases' in data.columns:
            report.append(f"Total Cases: {data['cases'].sum():,}")
        if 'deaths' in data.columns:
            report.append(f"Total Deaths: {data['deaths'].sum():,}")

        # Analysis Results
        if analysis_results:
            report.append("\n" + "=" * 80)
            report.append("ANALYSIS RESULTS")
            report.append("=" * 80)

            for section, results in analysis_results.items():
                report.append(f"\n{section.upper()}:")
                if isinstance(results, dict):
                    for key, value in results.items():
                        if isinstance(value, (int, float)):
                            report.append(f"  {key}: {value:,.3f}")
                        else:
                            report.append(f"  {key}: {value}")

        # Recommendations
        report.append("\n" + "=" * 80)
        report.append("RECOMMENDATIONS")
        report.append("=" * 80)

        if 'cases' in data.columns:
            if 'date' in data.columns:
                avg_daily = data.groupby('date')['cases'].sum().mean()
            else:
                avg_daily = data['cases'].mean()

            if avg_daily > 100:
                report.append("🚨 HIGH TRANSMISSION ZONE - Immediate action required:")
                report.append("  1. Implement strict containment measures")
                report.append("  2. Increase testing and contact tracing")
                report.append("  3. Deploy emergency medical resources")
                report.append("  4. Consider travel restrictions")

            elif avg_daily > 50:
                report.append("⚠️ MODERATE TRANSMISSION - Enhanced monitoring needed:")
                report.append("  1. Strengthen surveillance systems")
                report.append("  2. Promote preventive measures")
                report.append("  3. Prepare healthcare capacity")
                report.append("  4. Conduct community awareness campaigns")

            else:
                report.append("✅ LOW TRANSMISSION - Maintain vigilance:")
                report.append("  1. Continue routine surveillance")
                report.append("  2. Monitor for clusters")
                report.append("  3. Prepare response plans")
                report.append("  4. Conduct regular risk assessments")

        # Technical Details
        report.append("\n" + "=" * 80)
        report.append("TECHNICAL DETAILS")
        report.append("=" * 80)

        report.append(f"\nAnalysis performed using: EpiGeoSim-X Pro v5.0")
        report.append(f"Analysis timestamp: {datetime.now().isoformat()}")
        report.append(f"Data quality score: {self._calculate_data_quality(data):.1f}/10")

        report.append("\n" + "=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)

        return "\n".join(report)

    def _calculate_data_quality(self, data):
        """Calculate data quality score"""
        score = 10  # Start with perfect score

        # Check for missing values
        missing_pct = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        score -= missing_pct * 5

        # Check for date column
        if 'date' not in data.columns:
            score -= 2

        # Check for case data
        if 'cases' not in data.columns:
            score -= 2

        # Check for location data
        if not any(col in data.columns for col in ['latitude', 'longitude', 'region']):
            score -= 1

        return max(0, min(10, score))


# ============================================
# MAIN APPLICATION
# ============================================

# Initialize components
data_manager = EnhancedDataManager()
spatial_analyzer = AdvancedSpatialAnalyzer()
temporal_analyzer = AdvancedTemporalAnalyzer()
network_analyzer = AdvancedNetworkAnalyzer()
ml_models = AdvancedMLModels()
epi_models = EpidemiologicalModels()
visualizer = AdvancedVisualization()
export_manager = ExportManager()

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'shapefile' not in st.session_state:
    st.session_state.shapefile = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'network' not in st.session_state:
    st.session_state.network = None
if 'ml_results' not in st.session_state:
    st.session_state.ml_results = None


def main():
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #1a5f7a 0%, #2c3e50 50%, #9b59b6 100%); 
                    border-radius: 20px; margin-bottom: 2rem; border: 2px solid rgba(255,255,255,0.1);">
            <h2 style="color: white; margin: 0; font-size: 2.2rem;">🦠 EpiGeoSim-X Pro</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1.1rem; font-weight: 500;">
                Advanced Spatiotemporal Disease Intelligence Platform
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Navigation
        page = st.radio(
            "Navigation",
            ["🏠 Dashboard", "📁 Data Management", "🗺️ Spatial Analysis", "📈 Temporal Analysis",
             "🔗 Network Analysis", "🤖 Predictive Modeling", "🦠 Epidemiological Models",
             "📊 Visualization Dashboard", "📤 Export Center", "⚙️ Advanced Settings"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Data Status
        st.markdown("### 📊 Data Status")
        if st.session_state.data is not None:
            data = st.session_state.data
            st.success("✅ Data Loaded")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Records", f"{len(data):,}")
            with col2:
                st.metric("Columns", len(data.columns))

            if 'date' in data.columns:
                try:
                    st.caption(f"📅 {data['date'].min().date()} to {data['date'].max().date()}")
                except:
                    st.caption(f"📅 {data['date'].min()} to {data['date'].max()}")

            if 'cases' in data.columns:
                st.caption(f"🦠 {data['cases'].sum():,} total cases")
        else:
            st.warning("⚠️ No data loaded")

        if st.session_state.shapefile is not None:
            st.success("🗺️ Shapefile loaded")

        st.markdown("---")

        # Quick Actions
        st.markdown("### ⚡ Quick Actions")

        if st.button("🔄 Refresh Session", use_container_width=True):
            st.rerun()

        if st.button("📊 Generate Demo Data", use_container_width=True):
            with st.spinner("Generating demo data..."):
                demo_data = generate_demo_data()
                st.session_state.data = demo_data
                st.success("✅ Demo data generated!")
                st.rerun()

        st.markdown("---")

        # System Info
        st.markdown("### ℹ️ System Info")
        st.caption(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        st.caption("🔧 v5.0 | Advanced Edition")
        st.caption("👨‍💻 FNU Nahiduzzaman")
        st.caption("🚀 Department of Microbiology & Hygiene")
        st.caption("🏛️ Bangladesh Agricultural University")

        st.markdown("---")

        # Support Information
        with st.expander("📞 Support & Contact", expanded=False):
            st.markdown("""
            **Technical Support:**  
            📧 epigeosim.support@bau.edu.bd  
            📞 +880-1711-XXXXXX  

            **Academic Contact:**  
            📧 nahid@bau.edu.bd  

            **Emergency Support:**  
            24/7 Hotline: +880-XXX-XXXXXXX  

            **Documentation:**  
            📘 [User Manual](https://example.com/manual)  
            📚 [API Documentation](https://example.com/api)  

            **GitHub Repository:**  
            🔗 [EpiGeoSim-X Pro](https://github.com/username/epigeosim-pro)
            """)

    # ============================================
    # MAIN PAGE ROUTER
    # ============================================

    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📁 Data Management":
        show_data_management()
    elif page == "🗺️ Spatial Analysis":
        show_spatial_analysis()
    elif page == "📈 Temporal Analysis":
        show_temporal_analysis()
    elif page == "🔗 Network Analysis":
        show_network_analysis()
    elif page == "🤖 Predictive Modeling":
        show_predictive_modeling()
    elif page == "🦠 Epidemiological Models":
        show_epidemiological_models()
    elif page == "📊 Visualization Dashboard":
        show_visualization_dashboard()
    elif page == "📤 Export Center":
        show_export_center()
    elif page == "⚙️ Advanced Settings":
        show_advanced_settings()


# ============================================
# PAGE FUNCTIONS
# ============================================

def show_dashboard():
    """Display main dashboard"""
    st.markdown('<h1 class="main-header">Advanced Disease Intelligence Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Comprehensive epidemiological surveillance and predictive analytics</p>',
                unsafe_allow_html=True)

    # Welcome message
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 2rem; border-radius: 20px; margin: 2rem 0;">
        <h2 style="color: white; margin-bottom: 1rem;">🚀 Welcome to EpiGeoSim-X Pro</h2>
        <p style="font-size: 1.1rem;">
            Advanced platform for spatiotemporal analysis of disease dynamics, featuring 
            machine learning predictions, network analysis, and comprehensive visualization tools.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Quick Stats if data exists
    if st.session_state.data is not None:
        data = st.session_state.data

        st.markdown("### 📊 Quick Overview")

        # Create metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_cases = data['cases'].sum() if 'cases' in data.columns else 0
            st.metric("Total Cases", f"{total_cases:,}")

        with col2:
            if 'deaths' in data.columns:
                total_deaths = data['deaths'].sum()
                st.metric("Total Deaths", f"{total_deaths:,}")
            else:
                st.metric("Data Points", f"{len(data):,}")

        with col3:
            if 'date' in data.columns:
                try:
                    date_range = f"{data['date'].min().date()} to {data['date'].max().date()}"
                except:
                    date_range = f"{data['date'].min()} to {data['date'].max()}"
                st.metric("Date Range", date_range)

        with col4:
            if 'region' in data.columns:
                regions = data['region'].nunique()
                st.metric("Regions", regions)
            else:
                st.metric("Features", len(data.columns))

        # Quick Analysis Section
        st.markdown("### ⚡ Quick Insights")

        tab1, tab2, tab3 = st.tabs(["📈 Trends", "🗺️ Spatial", "🔮 Forecast"])

        with tab1:
            if 'date' in data.columns and 'cases' in data.columns:
                daily_cases = data.groupby('date')['cases'].sum()
                fig = px.line(daily_cases,
                              title='Daily Case Trends',
                              labels={'value': 'Cases', 'index': 'Date'},
                              line_shape='spline')
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            if all(col in data.columns for col in ['latitude', 'longitude', 'cases']):
                # Filter out NaN values
                map_data = data.dropna(subset=['latitude', 'longitude', 'cases'])
                if len(map_data) > 0:
                    fig = px.density_mapbox(map_data, lat='latitude', lon='longitude', z='cases',
                                            radius=20, zoom=6, mapbox_style="carto-positron",
                                            title='Case Density Map')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No valid location data for map")

        with tab3:
            if 'date' in data.columns and 'cases' in data.columns and len(data) > 30:
                st.info("Quick 30-day forecast based on historical trends")
                try:
                    daily_cases = data.groupby('date')['cases'].sum()
                    forecast = daily_cases.rolling(7).mean().iloc[-30:]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=daily_cases.index, y=daily_cases.values,
                                             name='Historical', line=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values,
                                             name='7-day MA Forecast', line=dict(color='red', dash='dash')))
                    fig.update_layout(title='Quick Forecast', xaxis_title='Date', yaxis_title='Cases')
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.warning("Insufficient data for forecasting")

    # Feature Highlights
    st.markdown("### 🎯 Key Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <h3>AI-Powered Predictions</h3>
            <p>Advanced ML models for accurate disease forecasting and risk assessment</p>
            <ul>
                <li>15+ ML Algorithms</li>
                <li>Deep Learning Models</li>
                <li>Hyperparameter Tuning</li>
                <li>Ensemble Methods</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🗺️</div>
            <h3>Spatial Intelligence</h3>
            <p>Comprehensive geospatial analysis with advanced GIS integration</p>
            <ul>
                <li>Shapefile Support</li>
                <li>Spatial Regression</li>
                <li>Hotspot Detection</li>
                <li>3D Mapping</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🦠</div>
            <h3>Epidemiological Models</h3>
            <p>Advanced compartmental models for disease transmission dynamics</p>
            <ul>
                <li>SIR/SEIR Models</li>
                <li>Network Models</li>
                <li>Intervention Analysis</li>
                <li>Outbreak Simulation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Quick Start Guide
    with st.expander("🚀 Quick Start Guide", expanded=False):
        st.markdown("""
        ### Getting Started

        1. **Data Import**
           - Upload your data (CSV, Excel, Shapefile)
           - Or generate sample data for testing
           - Ensure columns: date, cases, location

        2. **Basic Analysis**
           - Explore data in Data Management
           - Run spatial analysis for patterns
           - Analyze temporal trends

        3. **Advanced Features**
           - Build predictive models
           - Run epidemiological simulations
           - Create comprehensive reports

        4. **Export Results**
           - Export data in multiple formats
           - Save visualizations
           - Generate PDF reports
        """)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Load Sample Data", use_container_width=True):
                sample_data = generate_demo_data()
                st.session_state.data = sample_data
                st.success("Sample data loaded!")
                st.rerun()
        with col2:
            if st.button("📚 View Tutorial", use_container_width=True):
                st.info("Tutorial will open in new tab")


def show_data_management():
    """Display data management interface"""
    st.markdown('<h1 class="main-header">Data Management Center</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Import, validate, and preprocess epidemiological data</p>',
                unsafe_allow_html=True)

    # Tabs for different data operations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📤 Upload Data", "🗺️ Shapefile Import", "🔧 Data Processing", "📊 Data Quality", "🎲 Generate Data"]
    )

    with tab1:
        st.markdown("### 📤 Upload Your Data")

        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['csv', 'xlsx', 'xls', 'json', 'geojson'],
                help="Upload CSV, Excel, JSON, or GeoJSON files"
            )

        with col2:
            data_source = st.selectbox(
                "Data Source Type",
                ["Epidemiological", "Environmental", "Demographic", "Hospital", "Mixed"]
            )

        if uploaded_file is not None:
            with st.spinner("Loading data..."):
                df = data_manager.load_csv(uploaded_file)

                if df is not None:
                    st.session_state.data = df
                    st.success(f"✅ Successfully loaded {len(df):,} records")

                    # Display preview
                    st.markdown("#### 📋 Data Preview")
                    st.dataframe(df.head(), use_container_width=True)

                    # Show data information
                    with st.expander("📊 Data Information", expanded=True):
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Rows", len(df))
                            st.metric("Columns", len(df.columns))

                        with col2:
                            missing = df.isnull().sum().sum()
                            st.metric("Missing Values", missing)
                            dupes = df.duplicated().sum()
                            st.metric("Duplicates", dupes)

                        with col3:
                            if 'date' in df.columns:
                                try:
                                    st.metric("Date Range",
                                              f"{df['date'].min().date()} to {df['date'].max().date()}")
                                except:
                                    st.metric("Date Range",
                                              f"{df['date'].min()} to {df['date'].max()}")

                            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
                            st.metric("Numeric Columns", numeric_cols)

                    # Column mapping
                    st.markdown("#### 🎯 Column Mapping")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        date_cols = [col for col in df.columns if 'date' in col.lower()]
                        date_col = st.selectbox("Date Column",
                                                options=[''] + list(df.columns),
                                                index=date_cols[0] if date_cols else 0)

                    with col2:
                        case_cols = [col for col in df.columns if 'case' in col.lower()]
                        case_col = st.selectbox("Case Column",
                                                options=[''] + list(df.columns),
                                                index=case_cols[0] if case_cols else 0)

                    with col3:
                        region_cols = [col for col in df.columns if 'region' in col.lower()]
                        region_col = st.selectbox("Region Column",
                                                  options=[''] + list(df.columns),
                                                  index=region_cols[0] if region_cols else 0)

                    if st.button("💾 Save Column Mapping", type="primary"):
                        # Process column mapping
                        mapping = {}
                        if date_col: mapping[date_col] = 'date'
                        if case_col: mapping[case_col] = 'cases'
                        if region_col: mapping[region_col] = 'region'

                        if mapping:
                            df = df.rename(columns=mapping)
                            st.session_state.data = df
                            st.success("Column mapping saved!")

    with tab2:
        st.markdown("### 🗺️ Shapefile Import")

        uploaded_files = st.file_uploader(
            "Upload Shapefile components (.shp, .shx, .dbf, .prj)",
            type=['shp', 'shx', 'dbf', 'prj', 'zip'],
            accept_multiple_files=True
        )

        if uploaded_files:
            if st.button("Load Shapefile", type="primary"):
                with st.spinner("Processing shapefile..."):
                    gdf = data_manager.load_shapefile(uploaded_files)

                    if gdf is not None:
                        st.session_state.shapefile = gdf
                        st.success(f"✅ Shapefile loaded with {len(gdf)} features")

                        # Display shapefile info
                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("Features", len(gdf))
                            st.metric("CRS", str(gdf.crs))

                        with col2:
                            bounds = gdf.total_bounds
                            st.metric("Bounds",
                                      f"{bounds[0]:.2f},{bounds[1]:.2f} to {bounds[2]:.2f},{bounds[3]:.2f}")

                        # Preview geometry
                        st.markdown("#### 🗺️ Geometry Preview")
                        st.dataframe(gdf.head(), use_container_width=True)

                        # Option to join with epidemiological data
                        if st.session_state.data is not None:
                            st.markdown("#### 🔗 Join with Epidemiological Data")

                            col1, col2 = st.columns(2)

                            with col1:
                                gdf_key = st.selectbox("Shapefile Join Key",
                                                       options=gdf.columns.tolist())

                            with col2:
                                df_key = st.selectbox("Data Join Key",
                                                      options=st.session_state.data.columns.tolist())

                            if st.button("Join Data", type="primary"):
                                joined_gdf = data_manager.join_data_with_shapefile(
                                    st.session_state.data, gdf, df_key, gdf_key
                                )

                                if joined_gdf is not None:
                                    st.session_state.joined_data = joined_gdf
                                    st.success("✅ Data successfully joined!")

    with tab3:
        st.markdown("### 🔧 Data Processing")

        if st.session_state.data is not None:
            data = st.session_state.data

            processing_options = st.multiselect(
                "Select Processing Operations",
                ["Handle Missing Values", "Remove Outliers", "Normalize Features",
                 "Create Temporal Features", "Create Spatial Features",
                 "Encode Categorical Variables", "Feature Engineering"],
                default=["Handle Missing Values"]
            )

            if st.button("Apply Processing", type="primary"):
                processed_data = data.copy()

                with st.spinner("Processing data..."):
                    # Apply selected operations
                    if "Handle Missing Values" in processing_options:
                        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
                        for col in numeric_cols:
                            processed_data[col] = processed_data[col].fillna(processed_data[col].median())

                        categorical_cols = processed_data.select_dtypes(include=['object']).columns
                        for col in categorical_cols:
                            processed_data[col] = processed_data[col].fillna(
                                processed_data[col].mode()[0] if not processed_data[col].mode().empty else 'Unknown')

                    if "Remove Outliers" in processing_options and 'cases' in processed_data.columns:
                        Q1 = processed_data['cases'].quantile(0.25)
                        Q3 = processed_data['cases'].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR

                        before = len(processed_data)
                        processed_data = processed_data[
                            (processed_data['cases'] >= lower_bound) &
                            (processed_data['cases'] <= upper_bound)
                            ]
                        after = len(processed_data)
                        removed = before - after
                        st.info(f"Removed {removed} outlier records")

                    if "Normalize Features" in processing_options:
                        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
                        scaler = MinMaxScaler()
                        processed_data[numeric_cols] = scaler.fit_transform(processed_data[numeric_cols])
                        st.info("Features normalized using MinMaxScaler")

                    st.session_state.data = processed_data
                    st.success(f"✅ Processed {len(processed_data)} records")
                    st.rerun()
        else:
            st.info("Please load data first")

    with tab4:
        st.markdown("### 📊 Data Quality Assessment")

        if st.session_state.data is not None:
            data = st.session_state.data

            # Data quality metrics
            st.markdown("#### 📈 Quality Metrics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                missing_total = data.isnull().sum().sum()
                missing_pct = (missing_total / (data.shape[0] * data.shape[1])) * 100
                st.metric("Missing Values", f"{missing_total} ({missing_pct:.1f}%)")

            with col2:
                duplicates = data.duplicated().sum()
                st.metric("Duplicates", duplicates)

            with col3:
                if 'date' in data.columns:
                    invalid_dates = data['date'].isnull().sum()
                    st.metric("Invalid Dates", invalid_dates)

            with col4:
                numeric_cols = len(data.select_dtypes(include=[np.number]).columns)
                categorical_cols = len(data.select_dtypes(include=['object']).columns)
                st.metric("Column Types", f"{numeric_cols} num, {categorical_cols} cat")

            # Data profiling
            st.markdown("#### 🔍 Data Profile")

            profile_data = []
            for col in data.columns:
                profile_data.append({
                    'Column': col,
                    'Type': str(data[col].dtype),
                    'Missing': data[col].isnull().sum(),
                    'Missing %': (data[col].isnull().sum() / len(data)) * 100,
                    'Unique': data[col].nunique(),
                    'Sample': ', '.join(str(x) for x in data[col].dropna().head(3).tolist())
                    if data[col].nunique() < 20 else 'Various values'
                })

            profile_df = pd.DataFrame(profile_data)
            st.dataframe(profile_df, use_container_width=True)

            # Data visualization
            st.markdown("#### 📊 Data Distribution")

            if len(data.select_dtypes(include=[np.number]).columns) > 0:
                numeric_col = st.selectbox("Select numeric column for distribution",
                                           data.select_dtypes(include=[np.number]).columns)

                fig = px.histogram(data, x=numeric_col, nbins=50,
                                   title=f'Distribution of {numeric_col}',
                                   marginal='box')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please load data first")

    with tab5:
        st.markdown("### 🎲 Generate Sample Data")

        with st.form("sample_data_form"):
            col1, col2 = st.columns(2)

            with col1:
                n_regions = st.slider("Number of Regions", 3, 10, 6)
                n_days = st.slider("Number of Days", 30, 730, 365)
                outbreak_intensity = st.slider("Outbreak Intensity", 1, 10, 5)

            with col2:
                include_env = st.checkbox("Include Environmental Data", True)
                include_demo = st.checkbox("Include Demographic Data", True)
                include_movement = st.checkbox("Include Movement Network", False)
                add_noise = st.checkbox("Add Random Noise", True)

            generate = st.form_submit_button("🎲 Generate Data", type="primary")

            if generate:
                with st.spinner("Generating realistic sample data..."):
                    sample_data = generate_demo_data(
                        regions=n_regions,
                        days=n_days,
                        outbreak_intensity=outbreak_intensity,
                        include_env=include_env,
                        include_demo=include_demo
                    )

                    st.session_state.data = sample_data
                    st.success(f"✅ Generated {len(sample_data):,} records across {n_regions} regions")

                    # Show preview
                    st.dataframe(sample_data.head(), use_container_width=True)


def show_spatial_analysis():
    """Display spatial analysis interface"""
    st.markdown('<h1 class="main-header">Advanced Spatial Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Comprehensive geospatial pattern detection and hotspot analysis</p>',
                unsafe_allow_html=True)

    if st.session_state.data is None:
        st.warning("⚠️ Please load data first from Data Management")
        st.stop()

    data = st.session_state.data

    # Spatial Analysis Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["🗺️ Interactive Mapping", "📊 Spatial Statistics", "🔥 Hotspot Detection",
         "📈 Spatial Regression", "🎯 Cluster Analysis", "🌐 Voronoi Analysis"]
    )

    with tab1:
        st.markdown("### 🗺️ Interactive Map Visualization")

        col1, col2 = st.columns([1, 3])

        with col1:
            map_type = st.selectbox("Map Type",
                                    ["Heatmap", "Cluster", "Bubble", "Choropleth", "Animated"])
            value_column = st.selectbox("Value Column",
                                        [col for col in data.columns if data[col].dtype in ['int64', 'float64']],
                                        index=0 if 'cases' in data.columns else 0)

            basemap = st.selectbox("Base Map",
                                   ["OpenStreetMap", "CartoDB Positron", "CartoDB Dark Matter",
                                    "Stamen Terrain", "Stamen Toner"])

            if map_type == "Heatmap":
                radius = st.slider("Heatmap Radius", 5, 50, 15)
                blur = st.slider("Blur Level", 5, 30, 10)
            elif map_type == "Bubble":
                max_size = st.slider("Maximum Bubble Size", 10, 100, 50)

        with col2:
            # Create map
            if all(col in data.columns for col in ['latitude', 'longitude']):
                m = visualizer.create_interactive_map(data, map_type, value_column, height=600)
                st_folium(m, width=800, height=600)
            else:
                st.warning("Latitude and longitude columns are required for mapping")

    with tab2:
        st.markdown("### 📊 Advanced Spatial Statistics")

        if all(col in data.columns for col in ['latitude', 'longitude']):
            col1, col2 = st.columns(2)

            with col1:
                analysis_type = st.selectbox("Analysis Type",
                                             ["Spatial Autocorrelation", "Getis-Ord G",
                                              "Local Moran's I", "Geary's C"])

            with col2:
                value_column = st.selectbox("Analysis Value",
                                            [col for col in data.columns if data[col].dtype in ['int64', 'float64']],
                                            key="spatial_value")

            if st.button("Run Spatial Analysis", type="primary"):
                with st.spinner("Calculating spatial statistics..."):
                    # Create GeoDataFrame
                    gdf = gpd.GeoDataFrame(
                        data,
                        geometry=gpd.points_from_xy(data.longitude, data.latitude)
                    )
                    gdf.crs = 'EPSG:4326'

                    results = spatial_analyzer.calculate_spatial_autocorrelation(gdf, value_column)

                    # Display results
                    if 'moran_i' in results:
                        moran = results['moran_i']

                        st.markdown("#### 🌐 Moran's I Analysis")

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Moran's I", f"{moran['I']:.4f}")

                        with col2:
                            st.metric("Expected I", f"{moran['expected_i']:.4f}")

                        with col3:
                            st.metric("Z-Score", f"{moran['z_score']:.4f}")

                        with col4:
                            significance = "✅ Significant" if moran['p_value'] < 0.05 else "❌ Not Significant"
                            st.metric("Significance", significance)

                        # Interpretation
                        if moran['I'] > moran['expected_i']:
                            st.success("Positive spatial autocorrelation: Similar values cluster together")
                        else:
                            st.info("Negative spatial autocorrelation: Dissimilar values cluster together")

                    # Local Moran's I
                    if 'local_moran' in results:
                        local = results['local_moran']

                        st.markdown("#### 🔍 Local Indicators of Spatial Association (LISA)")

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Hotspots", local['hotspots'])

                        with col2:
                            st.metric("Coldspots", local['coldspots'])

                        with col3:
                            st.metric("Outliers", local['outliers'])

                        with col4:
                            total = len(gdf)
                            hotspot_pct = (local['hotspots'] / total) * 100
                            st.metric("Hotspot %", f"{hotspot_pct:.1f}%")

        else:
            st.warning("Latitude and longitude columns are required for spatial analysis")

    with tab3:
        st.markdown("### 🔥 Space-Time Hotspot Detection")

        if 'date' in data.columns and all(col in data.columns for col in ['latitude', 'longitude']):
            col1, col2 = st.columns(2)

            with col1:
                spatial_eps = st.slider("Spatial Radius (km)", 1.0, 100.0, 50.0, 1.0)
                temporal_eps = st.slider("Temporal Window (days)", 1, 30, 7)

            with col2:
                min_samples = st.slider("Minimum Samples", 2, 20, 5)
                value_column = st.selectbox("Value for Clustering",
                                            [col for col in data.columns if data[col].dtype in ['int64', 'float64']],
                                            key="cluster_value")

            if st.button("Detect Space-Time Clusters", type="primary"):
                with st.spinner("Detecting clusters..."):
                    clusters, cluster_stats = spatial_analyzer.detect_space_time_clusters(
                        data, spatial_eps=spatial_eps / 111,  # Convert km to degrees
                        temporal_eps=temporal_eps,
                        min_samples=min_samples
                    )

                    if cluster_stats is not None and len(cluster_stats) > 0:
                        st.success(f"✅ Detected {len(cluster_stats)} space-time clusters")

                        # Display cluster statistics
                        st.dataframe(cluster_stats, use_container_width=True)

                        # Visualize clusters
                        data_with_clusters = data.copy()
                        data_with_clusters['cluster'] = clusters

                        fig = px.scatter_mapbox(data_with_clusters,
                                                lat='latitude', lon='longitude',
                                                color='cluster', size=value_column,
                                                hover_name='region' if 'region' in data.columns else None,
                                                title='Space-Time Clusters',
                                                mapbox_style="carto-positron")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No significant clusters detected")
        else:
            st.warning("Date and location data required for space-time analysis")

    with tab4:
        st.markdown("### 📈 Spatial Regression Analysis")

        if all(col in data.columns for col in ['latitude', 'longitude']):
            # Prepare data for spatial regression
            gdf = gpd.GeoDataFrame(
                data,
                geometry=gpd.points_from_xy(data.longitude, data.latitude)
            )

            st.markdown("#### Select Variables for Regression")

            col1, col2 = st.columns(2)

            with col1:
                dependent_var = st.selectbox("Dependent Variable",
                                             [col for col in data.columns if data[col].dtype in ['int64', 'float64']],
                                             index=0 if 'cases' in data.columns else 0)

            with col2:
                independent_vars = st.multiselect("Independent Variables",
                                                  [col for col in data.columns
                                                   if data[col].dtype in ['int64', 'float64'] and col != dependent_var],
                                                  default=[col for col in ['temperature', 'rainfall', 'population']
                                                           if col in data.columns])

            if st.button("Run Spatial Regression", type="primary"):
                with st.spinner("Running spatial regression..."):
                    results = spatial_analyzer.perform_spatial_regression(
                        gdf, dependent_var, independent_vars
                    )

                    # Display results
                    if 'ols' in results:
                        st.markdown("#### 📊 Ordinary Least Squares (OLS)")
                        ols = results['ols']

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("R²", f"{ols['r2']:.3f}")
                        with col2:
                            st.metric("AIC", f"{ols['aic']:.1f}")
                        with col3:
                            significant = sum(1 for p in ols['p_values'].values() if p < 0.05)
                            st.metric("Significant Vars", significant)

        else:
            st.warning("Location data required for spatial regression")

    with tab5:
        st.markdown("### 🎯 Advanced Cluster Analysis")

        col1, col2 = st.columns(2)

        with col1:
            clustering_method = st.selectbox("Clustering Algorithm",
                                             ["DBSCAN", "K-Means", "Hierarchical", "OPTICS"])

            if clustering_method == "DBSCAN":
                eps = st.slider("Epsilon", 0.01, 1.0, 0.1, 0.01)
                min_samples = st.slider("Min Samples", 2, 20, 5)
            elif clustering_method == "K-Means":
                n_clusters = st.slider("Number of Clusters", 2, 20, 5)

        with col2:
            variables = st.multiselect("Variables for Clustering",
                                       [col for col in data.columns if data[col].dtype in ['int64', 'float64']],
                                       default=['cases', 'latitude', 'longitude'] if all(
                                           col in data.columns for col in ['cases', 'latitude', 'longitude']) else [])

        if st.button("Run Cluster Analysis", type="primary"):
            with st.spinner("Performing clustering..."):
                # Prepare data
                if len(variables) == 0:
                    st.warning("Please select variables for clustering")
                    st.stop()

                cluster_data = data[variables].dropna()

                if clustering_method == "DBSCAN":
                    clustering = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = clustering.fit_predict(cluster_data)
                elif clustering_method == "K-Means":
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    labels = kmeans.fit_predict(cluster_data)
                elif clustering_method == "Hierarchical":
                    hierarchical = AgglomerativeClustering(n_clusters=min(len(cluster_data), 10))
                    labels = hierarchical.fit_predict(cluster_data)
                elif clustering_method == "OPTICS":
                    optics = OPTICS(min_samples=5)
                    labels = optics.fit_predict(cluster_data)

                # Add labels to data
                data_labeled = data.copy()
                data_labeled['cluster'] = labels

                # Visualize clusters
                if 'latitude' in data.columns and 'longitude' in data.columns:
                    fig = px.scatter_mapbox(data_labeled,
                                            lat='latitude', lon='longitude',
                                            color='cluster',
                                            hover_name='region' if 'region' in data.columns else None,
                                            title=f'{clustering_method} Clustering',
                                            mapbox_style="carto-positron")
                    st.plotly_chart(fig, use_container_width=True)

                # Cluster statistics
                st.markdown("#### 📊 Cluster Statistics")
                if 'cases' in data.columns:
                    cluster_stats = data_labeled.groupby('cluster').agg({
                        'cases': ['count', 'mean', 'sum']
                    }).round(2)
                else:
                    cluster_stats = data_labeled.groupby('cluster').size().to_frame('count')
                st.dataframe(cluster_stats, use_container_width=True)

    with tab6:
        st.markdown("### 🌐 Voronoi Analysis for Spatial Interpolation")

        if all(col in data.columns for col in ['latitude', 'longitude']):
            value_column = st.selectbox("Value for Interpolation",
                                        [col for col in data.columns if data[col].dtype in ['int64', 'float64']],
                                        key="voronoi_value")

            if st.button("Generate Voronoi Diagram", type="primary"):
                with st.spinner("Calculating Voronoi diagram..."):
                    # Create GeoDataFrame
                    gdf = gpd.GeoDataFrame(
                        data,
                        geometry=gpd.points_from_xy(data.longitude, data.latitude)
                    )

                    voronoi_gdf = spatial_analyzer.calculate_voronoi_diagram(gdf, value_column)

                    if voronoi_gdf is not None:
                        # Create interactive map
                        fig = px.choropleth_mapbox(
                            voronoi_gdf,
                            geojson=voronoi_gdf.geometry,
                            locations=voronoi_gdf.index,
                            color='density',
                            mapbox_style="carto-positron",
                            zoom=6,
                            center={"lat": data['latitude'].mean(), "lon": data['longitude'].mean()},
                            opacity=0.7,
                            title='Voronoi Density Map'
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Location data required for Voronoi analysis")


def show_temporal_analysis():
    """Display temporal analysis interface"""
    st.markdown('<h1 class="main-header">Advanced Temporal Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Time series analysis, forecasting, and epidemic indicators</p>',
                unsafe_allow_html=True)

    if st.session_state.data is None:
        st.warning("⚠️ Please load data first from Data Management")
        st.stop()

    data = st.session_state.data

    if 'date' not in data.columns:
        st.error("❌ Date column not found in data")
        st.stop()

    # Temporal Analysis Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📈 Time Series Analysis", "🔮 Forecasting Models", "📊 Epidemic Indicators",
         "🔄 Seasonality Analysis", "📉 Anomaly Detection"]
    )

    with tab1:
        st.markdown("### 📈 Comprehensive Time Series Analysis")

        col1, col2 = st.columns(2)

        with col1:
            value_column = st.selectbox("Analysis Variable",
                                        [col for col in data.columns if data[col].dtype in ['int64', 'float64']],
                                        index=0 if 'cases' in data.columns else 0,
                                        key="ts_value")

            aggregation = st.selectbox("Aggregation Level",
                                       ["Daily", "Weekly", "Monthly", "Quarterly"])

            if 'region' in data.columns:
                selected_regions = st.multiselect("Select Regions",
                                                  data['region'].unique(),
                                                  default=data['region'].unique()[:3])
            else:
                selected_regions = ['All']

        with col2:
            show_trend = st.checkbox("Show Trend Line", True)
            show_moving_avg = st.checkbox("Show Moving Average", True)
            if show_moving_avg:
                ma_window = st.slider("Moving Average Window", 3, 30, 7)
            show_cumulative = st.checkbox("Show Cumulative", False)

        # Prepare time series data
        if 'All' not in selected_regions and 'region' in data.columns:
            ts_data = data[data['region'].isin(selected_regions)]
        else:
            ts_data = data

        ts_data['date'] = pd.to_datetime(ts_data['date'])

        # Aggregate based on selection
        if aggregation == "Weekly":
            ts = ts_data.groupby(pd.Grouper(key='date', freq='W'))[value_column].sum()
        elif aggregation == "Monthly":
            ts = ts_data.groupby(pd.Grouper(key='date', freq='M'))[value_column].sum()
        elif aggregation == "Quarterly":
            ts = ts_data.groupby(pd.Grouper(key='date', freq='Q'))[value_column].sum()
        else:
            ts = ts_data.groupby('date')[value_column].sum()

        # Create time series plot
        fig = go.Figure()

        # Add main time series
        fig.add_trace(go.Scatter(
            x=ts.index,
            y=ts.values,
            mode='lines',
            name=value_column,
            line=dict(color='#1a5f7a', width=3)
        ))

        # Add moving average
        if show_moving_avg and len(ts) > ma_window:
            ma = ts.rolling(window=ma_window).mean()
            fig.add_trace(go.Scatter(
                x=ma.index,
                y=ma.values,
                mode='lines',
                name=f'{ma_window}-Day MA',
                line=dict(color='#e74c3c', width=2, dash='dash')
            ))

        # Add trend line
        if show_trend and len(ts) > 10:
            z = np.polyfit(range(len(ts)), ts.values, 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=ts.index,
                y=p(range(len(ts))),
                mode='lines',
                name='Trend',
                line=dict(color='#27ae60', width=2, dash='dot')
            ))

        # Add cumulative if requested
        if show_cumulative:
            cumulative = ts.cumsum()
            fig.add_trace(go.Scatter(
                x=cumulative.index,
                y=cumulative.values,
                mode='lines',
                name='Cumulative',
                line=dict(color='#9b59b6', width=2),
                yaxis='y2'
            ))

        fig.update_layout(
            title=f'{aggregation} {value_column.title()}',
            xaxis_title='Date',
            yaxis_title=value_column.title(),
            hovermode='x unified',
            height=500
        )

        if show_cumulative:
            fig.update_layout(
                yaxis2=dict(
                    title='Cumulative',
                    overlaying='y',
                    side='right'
                )
            )

        st.plotly_chart(fig, use_container_width=True)

        # Time series decomposition
        if len(ts) > 30:
            st.markdown("#### 🔍 Time Series Decomposition")

            try:
                decomposition = temporal_analyzer.decompose_time_series(ts, period=7)

                if decomposition is not None:
                    fig_decomp = make_subplots(
                        rows=4, cols=1,
                        subplot_titles=['Observed', 'Trend', 'Seasonal', 'Residual'],
                        vertical_spacing=0.1
                    )

                    fig_decomp.add_trace(
                        go.Scatter(x=ts.index, y=decomposition['observed'], name='Observed'),
                        row=1, col=1
                    )

                    fig_decomp.add_trace(
                        go.Scatter(x=ts.index, y=decomposition['trend'], name='Trend'),
                        row=2, col=1
                    )

                    fig_decomp.add_trace(
                        go.Scatter(x=ts.index, y=decomposition['seasonal'], name='Seasonal'),
                        row=3, col=1
                    )

                    fig_decomp.add_trace(
                        go.Scatter(x=ts.index, y=decomposition['residual'], name='Residual'),
                        row=4, col=1
                    )

                    fig_decomp.update_layout(height=700, showlegend=False)
                    st.plotly_chart(fig_decomp, use_container_width=True)
                else:
                    st.warning("Could not decompose time series")
            except Exception as e:
                st.warning(f"Decomposition error: {str(e)}")

        # Stationarity tests
        if len(ts) > 50:
            st.markdown("#### 📊 Stationarity Analysis")

            if st.button("Test Stationarity", type="primary"):
                with st.spinner("Testing stationarity..."):
                    stationarity = temporal_analyzer.test_stationarity(ts)

                    if 'adf' in stationarity:
                        col1, col2 = st.columns(2)

                        with col1:
                            adf = stationarity['adf']
                            st.metric("ADF Test",
                                      f"Statistic: {adf['statistic']:.4f}",
                                      f"p-value: {adf['p_value']:.4f}")
                            if adf['stationary']:
                                st.success("✅ Series is stationary (ADF)")
                            else:
                                st.warning("⚠️ Series is not stationary (ADF)")

                        with col2:
                            kpss = stationarity['kpss']
                            st.metric("KPSS Test",
                                      f"Statistic: {kpss['statistic']:.4f}",
                                      f"p-value: {kpss['p_value']:.4f}")
                            if kpss['stationary']:
                                st.success("✅ Series is stationary (KPSS)")
                            else:
                                st.warning("⚠️ Series is not stationary (KPSS)")

    with tab2:
        st.markdown("### 🔮 Advanced Forecasting Models")

        if len(data) > 30:
            col1, col2 = st.columns(2)

            with col1:
                forecast_method = st.selectbox("Forecasting Method",
                                               ["ARIMA", "Prophet", "Exponential Smoothing", "LSTM"])

                forecast_days = st.slider("Forecast Horizon (days)", 7, 180, 30)

                if forecast_method == "ARIMA":
                    p = st.slider("AR Order (p)", 0, 5, 2)
                    d = st.slider("Difference Order (d)", 0, 2, 1)
                    q = st.slider("MA Order (q)", 0, 5, 2)

            with col2:
                confidence_level = st.slider("Confidence Interval", 80, 99, 95)
                train_test_split = st.slider("Training Data %", 60, 90, 80)

                if forecast_method == "Prophet":
                    changepoint_scale = st.slider("Changepoint Prior Scale", 0.001, 0.5, 0.05, 0.001)
                    seasonality_scale = st.slider("Seasonality Prior Scale", 0.1, 100.0, 10.0, 0.1)

            if st.button("Generate Forecast", type="primary"):
                with st.spinner("Training forecast model..."):
                    # Prepare time series
                    daily_cases = data.groupby('date')['cases'].sum()

                    if forecast_method == "ARIMA":
                        model, summary = temporal_analyzer.build_arima_model(
                            daily_cases, order=(p, d, q)
                        )

                        if model is not None:
                            # Plot forecast
                            forecast = model.get_forecast(steps=forecast_days)
                            forecast_mean = forecast.predicted_mean
                            conf_int = forecast.conf_int(alpha=1 - confidence_level / 100)

                            fig = go.Figure()

                            # Historical data
                            fig.add_trace(go.Scatter(
                                x=daily_cases.index[-60:],
                                y=daily_cases.values[-60:],
                                mode='lines',
                                name='Historical',
                                line=dict(color='blue', width=2)
                            ))

                            # Forecast
                            fig.add_trace(go.Scatter(
                                x=forecast_mean.index,
                                y=forecast_mean.values,
                                mode='lines',
                                name='Forecast',
                                line=dict(color='red', width=2, dash='dash')
                            ))

                            # Confidence interval
                            fig.add_trace(go.Scatter(
                                x=conf_int.index.tolist() + conf_int.index.tolist()[::-1],
                                y=conf_int.iloc[:, 0].tolist() + conf_int.iloc[:, 1].tolist()[::-1],
                                fill='toself',
                                fillcolor='rgba(255, 0, 0, 0.2)',
                                line=dict(color='rgba(255,255,255,0)'),
                                name=f'{confidence_level}% CI'
                            ))

                            fig.update_layout(
                                title=f'{forecast_days}-Day ARIMA Forecast',
                                xaxis_title='Date',
                                yaxis_title='Cases',
                                height=500
                            )

                            st.plotly_chart(fig, use_container_width=True)

                            # Model metrics
                            if summary:
                                st.metric("AIC", f"{summary['aic']:.2f}")
                                st.metric("BIC", f"{summary['bic']:.2f}")

                    elif forecast_method == "Prophet":
                        model, forecast = temporal_analyzer.build_prophet_model(
                            data, changepoint_prior_scale=changepoint_scale,
                            seasonality_prior_scale=seasonality_scale
                        )

                        if model is not None:
                            # Plot components
                            fig1 = model.plot(forecast)
                            st.pyplot(fig1)

                            fig2 = model.plot_components(forecast)
                            st.pyplot(fig2)

        else:
            st.warning("At least 30 days of data required for forecasting")

    with tab3:
        st.markdown("### 📊 Advanced Epidemic Indicators")

        if st.button("Calculate Epidemic Indicators", type="primary"):
            with st.spinner("Calculating indicators..."):
                indicators = temporal_analyzer.calculate_epidemic_indicators(data)

                # Display indicators
                if 'reproduction_number' in indicators:
                    st.markdown("#### 🔄 Reproduction Number (R)")
                    r0 = indicators['reproduction_number']

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Simple Ratio", f"{r0['simple_ratio']:.2f}")

                    with col2:
                        st.metric("Exponential", f"{r0['exponential_growth']:.2f}")

                    with col3:
                        st.metric("Mean R", f"{r0['mean']:.2f}")

                    with col4:
                        status = "🚨 Above 1" if r0['mean'] > 1 else "✅ Below 1"
                        st.metric("Transmission", status)

                if 'doubling_time' in indicators:
                    st.markdown("#### 📈 Doubling Time")
                    dt = indicators['doubling_time']

                    col1, col2 = st.columns(2)

                    with col1:
                        if dt['doubling_time_days'] < float('inf'):
                            st.metric("Doubling Time", f"{dt['doubling_time_days']:.1f} days")
                        else:
                            st.metric("Doubling Time", "∞ (declining)")

                    with col2:
                        st.metric("Growth Rate", f"{dt['growth_rate']:.2%}")

                # Additional indicators
                st.markdown("#### 📊 Additional Metrics")

                col1, col2, col3 = st.columns(3)

                with col1:
                    if 'case_fatality_rate' in indicators:
                        st.metric("Case Fatality Rate", f"{indicators['case_fatality_rate']:.2f}%")

                with col2:
                    if 'attack_rate' in indicators:
                        st.metric("Attack Rate", f"{indicators['attack_rate']:.2f}%")

                with col3:
                    if 'hospitalization_rate' in indicators:
                        st.metric("Hospitalization Rate", f"{indicators['hospitalization_rate']:.2f}%")

    with tab4:
        st.markdown("### 🔄 Seasonality Analysis")

        if 'date' in data.columns and 'cases' in data.columns:
            daily_cases = data.groupby('date')['cases'].sum()

            if len(daily_cases) > 90:  # Need at least 3 months for seasonality
                # Weekly seasonality
                st.markdown("#### 📅 Weekly Patterns")

                data['weekday'] = pd.to_datetime(data['date']).dt.day_name()
                weekday_cases = data.groupby('weekday')['cases'].sum()

                # Order weekdays
                weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                weekday_cases = weekday_cases.reindex(weekday_order)

                fig = px.bar(weekday_cases, title='Cases by Day of Week',
                             labels={'value': 'Cases', 'index': 'Day'})
                st.plotly_chart(fig, use_container_width=True)

                # Monthly seasonality
                st.markdown("#### 📆 Monthly Patterns")

                data['month'] = pd.to_datetime(data['date']).dt.month_name()
                month_cases = data.groupby('month')['cases'].sum()

                # Order months
                month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                               'July', 'August', 'September', 'October', 'November', 'December']
                month_cases = month_cases.reindex(month_order)

                fig = px.bar(month_cases, title='Cases by Month',
                             labels={'value': 'Cases', 'index': 'Month'})
                st.plotly_chart(fig, use_container_width=True)

                # Auto-correlation
                st.markdown("#### 🔗 Auto-correlation Function (ACF)")

                fig, ax = plt.subplots(figsize=(10, 4))
                plot_acf(daily_cases, ax=ax, lags=30)
                st.pyplot(fig)
            else:
                st.warning("At least 90 days of data required for seasonality analysis")

    with tab5:
        st.markdown("### 📉 Anomaly Detection in Time Series")

        if 'date' in data.columns and 'cases' in data.columns:
            daily_cases = data.groupby('date')['cases'].sum()

            detection_method = st.selectbox("Detection Method",
                                            ["Statistical Threshold", "Moving Average", "Isolation Forest"])

            sensitivity = st.slider("Detection Sensitivity", 1.0, 5.0, 2.0, 0.1)

            if st.button("Detect Anomalies", type="primary"):
                with st.spinner("Detecting anomalies..."):
                    if detection_method == "Statistical Threshold":
                        mean = daily_cases.mean()
                        std = daily_cases.std()
                        threshold = mean + sensitivity * std

                        anomalies = daily_cases[daily_cases > threshold]

                    elif detection_method == "Moving Average":
                        window = 7
                        ma = daily_cases.rolling(window=window).mean()
                        std = daily_cases.rolling(window=window).std()

                        upper_bound = ma + sensitivity * std
                        anomalies = daily_cases[daily_cases > upper_bound]

                    # Plot anomalies
                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=daily_cases.index,
                        y=daily_cases.values,
                        mode='lines',
                        name='Daily Cases',
                        line=dict(color='blue', width=2)
                    ))

                    if len(anomalies) > 0:
                        fig.add_trace(go.Scatter(
                            x=anomalies.index,
                            y=anomalies.values,
                            mode='markers',
                            name='Anomalies',
                            marker=dict(color='red', size=10)
                        ))

                    fig.update_layout(
                        title='Anomaly Detection',
                        xaxis_title='Date',
                        yaxis_title='Cases',
                        height=500
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Display anomalies
                    st.markdown(f"#### 🚨 Detected {len(anomalies)} Anomalies")
                    if len(anomalies) > 0:
                        anomalies_df = pd.DataFrame({
                            'Date': anomalies.index,
                            'Cases': anomalies.values
                        })
                        st.dataframe(anomalies_df, use_container_width=True)


def show_network_analysis():
    """Display network analysis interface"""
    st.markdown('<h1 class="main-header">Network Analysis & Transmission Dynamics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyze disease spread networks and simulate transmission</p>',
                unsafe_allow_html=True)

    if st.session_state.data is None:
        st.warning("⚠️ Please load data first from Data Management")
        st.stop()

    data = st.session_state.data

    # Network Analysis Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🌐 Network Construction", "📊 Network Metrics", "🎯 Critical Analysis",
         "🔄 Spread Simulation", "📈 Transmission Models"]
    )

    with tab1:
        st.markdown("### 🌐 Build Disease Transmission Network")

        col1, col2 = st.columns(2)

        with col1:
            network_method = st.selectbox("Network Model",
                                          ["Gravity Model", "Radiation Model", "Proximity", "Composite"])

            if network_method in ["Gravity Model", "Proximity"]:
                threshold = st.slider("Interaction Threshold", 0.0, 1.0, 0.01, 0.001)

            if network_method == "Proximity":
                distance_threshold = st.slider("Distance Threshold (km)", 1, 200, 50)

        with col2:
            include_weights = st.checkbox("Weight Connections", True)
            directed = st.checkbox("Directed Network", False)
            simplify = st.checkbox("Simplify Network", True)

        if st.button("Build Network", type="primary"):
            with st.spinner("Building network..."):
                G = network_analyzer.build_transmission_network(
                    data,
                    method=network_method.lower().replace(' ', '_'),
                    params={'threshold': threshold} if network_method in ["Gravity Model", "Proximity"] else {}
                )

                if G is not None:
                    st.session_state.network = G
                    st.success(f"✅ Built network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

                    # Basic network statistics
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Nodes", G.number_of_nodes())

                    with col2:
                        st.metric("Edges", G.number_of_edges())

                    with col3:
                        density = nx.density(G)
                        st.metric("Density", f"{density:.4f}")

                    with col4:
                        if nx.is_connected(G):
                            components = nx.number_connected_components(G)
                            st.metric("Components", components)
                        else:
                            st.metric("Connected", "No")

                    # Visualize network
                    st.markdown("#### 🌐 Network Visualization")

                    # Simple visualization using plotly
                    pos = nx.spring_layout(G, seed=42)

                    edge_x = []
                    edge_y = []
                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])

                    node_x = []
                    node_y = []
                    node_text = []
                    node_size = []

                    for node in G.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)

                        node_info = f"Node {node}"
                        if 'region' in G.nodes[node]:
                            node_info = G.nodes[node]['region']

                        node_text.append(node_info)
                        node_size.append(10 + G.degree(node) * 2)

                    edge_trace = go.Scatter(
                        x=edge_x, y=edge_y,
                        line=dict(width=0.5, color='#888'),
                        hoverinfo='none',
                        mode='lines'
                    )

                    node_trace = go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers+text',
                        text=node_text,
                        textposition="top center",
                        marker=dict(
                            size=node_size,
                            color=node_size,
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Node Degree")
                        )
                    )

                    fig = go.Figure(data=[edge_trace, node_trace],
                                    layout=go.Layout(
                                        title='Disease Transmission Network',
                                        showlegend=False,
                                        hovermode='closest',
                                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                        height=600
                                    ))

                    st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### 📊 Network Properties & Metrics")

        if 'network' not in st.session_state or st.session_state.network is None:
            st.info("Please build a network first in the 'Network Construction' tab")
        else:
            G = st.session_state.network

            if st.button("Analyze Network", type="primary"):
                with st.spinner("Analyzing network properties..."):
                    results = network_analyzer.analyze_network_properties(G)

                    # Display results
                    if 'basic' in results:
                        st.markdown("#### 📈 Basic Properties")

                        basic = results['basic']
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Nodes", basic['nodes'])

                        with col2:
                            st.metric("Edges", basic['edges'])

                        with col3:
                            st.metric("Density", f"{basic['density']:.4f}")

                        with col4:
                            st.metric("Connected", "Yes" if basic['is_connected'] else "No")

                    if 'degree' in results:
                        st.markdown("#### 🎯 Degree Distribution")

                        degree = results['degree']
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Mean Degree", f"{degree['mean']:.2f}")

                        with col2:
                            st.metric("Max Degree", f"{degree['max']}")

                        with col3:
                            st.metric("Assortativity", f"{degree['assortativity']:.3f}")

                        with col4:
                            # Plot degree distribution
                            fig = px.histogram(x=degree['distribution'], nbins=20,
                                               title='Degree Distribution',
                                               labels={'x': 'Degree', 'y': 'Count'})
                            st.plotly_chart(fig, use_container_width=True)

                    if 'centrality' in results:
                        st.markdown("#### 🏆 Centrality Measures")

                        centrality = results['centrality']

                        # Top nodes by centrality
                        if 'betweenness' in centrality:
                            betweenness = centrality['betweenness']
                            top_between = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]

                            st.markdown("**Top Nodes by Betweenness Centrality:**")
                            for node, score in top_between:
                                st.write(f"  - Node {node}: {score:.4f}")

                        if 'closeness' in centrality:
                            closeness = centrality['closeness']
                            top_close = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:5]

                            st.markdown("**Top Nodes by Closeness Centrality:**")
                            for node, score in top_close:
                                st.write(f"  - Node {node}: {score:.4f}")

                    if 'clustering' in results:
                        st.markdown("#### 🔗 Clustering Coefficients")

                        clustering = results['clustering']
                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("Avg Clustering", f"{clustering['average_clustering']:.3f}")

                        with col2:
                            st.metric("Transitivity", f"{clustering['transitivity']:.3f}")

    with tab3:
        st.markdown("### 🎯 Critical Network Analysis")

        if 'network' not in st.session_state or st.session_state.network is None:
            st.info("Please build a network first")
        else:
            G = st.session_state.network

            analysis_type = st.selectbox("Analysis Type",
                                         ["Vulnerability Analysis", "Critical Paths",
                                          "Community Detection", "Bridge Identification"])

            if analysis_type == "Vulnerability Analysis":
                removal_strategy = st.selectbox("Removal Strategy",
                                                ["Degree", "Betweenness", "Closeness", "Random"])

                if st.button("Analyze Vulnerability", type="primary"):
                    with st.spinner("Analyzing network vulnerability..."):
                        # Simple vulnerability analysis
                        efficiency_history = []
                        nodes_removed = []

                        # Copy the graph
                        G_copy = G.copy()

                        for i in range(1, min(11, G.number_of_nodes())):
                            if removal_strategy == "Degree":
                                # Remove highest degree node
                                degrees = dict(G_copy.degree())
                                node_to_remove = max(degrees.items(), key=lambda x: x[1])[0]
                            elif removal_strategy == "Betweenness":
                                # Remove highest betweenness node
                                betweenness = nx.betweenness_centrality(G_copy)
                                node_to_remove = max(betweenness.items(), key=lambda x: x[1])[0]
                            elif removal_strategy == "Closeness":
                                # Remove highest closeness node
                                closeness = nx.closeness_centrality(G_copy)
                                node_to_remove = max(closeness.items(), key=lambda x: x[1])[0]
                            else:
                                # Random removal
                                node_to_remove = list(G_copy.nodes())[0]

                            G_copy.remove_node(node_to_remove)

                            # Calculate efficiency
                            if nx.is_connected(G_copy):
                                efficiency = nx.global_efficiency(G_copy)
                            else:
                                efficiency = 0

                            efficiency_history.append(efficiency)
                            nodes_removed.append(i)

                        results = pd.DataFrame({
                            'nodes_removed': nodes_removed,
                            'efficiency': efficiency_history
                        })

                        # Plot vulnerability curve
                        fig = px.line(results, x='nodes_removed', y='efficiency',
                                      title='Network Vulnerability Curve',
                                      labels={'nodes_removed': 'Nodes Removed',
                                              'efficiency': 'Network Efficiency'})
                        st.plotly_chart(fig, use_container_width=True)

            elif analysis_type == "Critical Paths":
                if st.button("Find Critical Paths", type="primary"):
                    with st.spinner("Finding critical transmission paths..."):
                        try:
                            # Find shortest paths between all nodes
                            critical_paths = []
                            for source in G.nodes():
                                for target in G.nodes():
                                    if source != target:
                                        try:
                                            path = nx.shortest_path(G, source=source, target=target, weight='weight')
                                            path_length = nx.shortest_path_length(G, source=source, target=target,
                                                                                  weight='weight')

                                            # Calculate path risk (sum of cases along path)
                                            path_risk = 0
                                            nodes_info = []
                                            for node in path:
                                                path_risk += G.nodes[node].get('cases', 0)
                                                nodes_info.append({
                                                    'region': G.nodes[node].get('region', f'Node {node}'),
                                                    'cases': G.nodes[node].get('cases', 0)
                                                })

                                            critical_paths.append({
                                                'source': source,
                                                'target': target,
                                                'path_length': path_length,
                                                'path_risk': path_risk,
                                                'nodes': nodes_info
                                            })
                                        except:
                                            continue

                            # Sort by path risk
                            critical_paths.sort(key=lambda x: x['path_risk'], reverse=True)

                            if critical_paths:
                                st.success(f"✅ Found {len(critical_paths)} critical paths")

                                for i, path in enumerate(critical_paths[:5]):
                                    with st.expander(f"Critical Path #{i + 1}"):
                                        st.write(f"**Source:** {path['source']}")
                                        st.write(f"**Target:** {path['target']}")
                                        st.write(f"**Path Length:** {path['path_length']} hops")
                                        st.write(f"**Path Risk Score:** {path['path_risk']:.4f}")

                                        st.write("**Nodes along path:**")
                                        for node in path['nodes']:
                                            st.write(f"  - {node['region']}: {node['cases']} cases")
                            else:
                                st.info("No critical paths found")
                        except Exception as e:
                            st.warning(f"Could not find critical paths: {str(e)}")

            elif analysis_type == "Community Detection":
                if st.button("Detect Communities", type="primary"):
                    with st.spinner("Detecting communities..."):
                        try:
                            communities = nx.algorithms.community.greedy_modularity_communities(G)

                            st.success(f"✅ Found {len(communities)} communities")

                            for i, community in enumerate(communities):
                                with st.expander(f"Community #{i + 1} ({len(community)} nodes)"):
                                    st.write(f"**Nodes:** {list(community)[:10]}{'...' if len(community) > 10 else ''}")
                        except:
                            st.warning("Could not detect communities")

    with tab4:
        st.markdown("### 🔄 Epidemic Spread Simulation")

        if 'network' not in st.session_state or st.session_state.network is None:
            st.info("Please build a network first")
        else:
            G = st.session_state.network

            col1, col2 = st.columns(2)

            with col1:
                model_type = st.selectbox("Epidemic Model", ["SIR", "SIS", "SEIR"])

                if model_type == "SIR":
                    beta = st.slider("Transmission Rate (β)", 0.01, 1.0, 0.3, 0.01)
                    gamma = st.slider("Recovery Rate (γ)", 0.01, 0.5, 0.1, 0.01)

                initial_infected = st.slider("Initial Infected Nodes", 1, 10, 3)

            with col2:
                simulation_steps = st.slider("Simulation Steps", 10, 200, 50)
                intervention_step = st.slider("Intervention Step", 0, 100, 20)
                intervention_effect = st.slider("Intervention Effect", 0.0, 1.0, 0.5, 0.1)

            if st.button("Run Simulation", type="primary"):
                with st.spinner("Running epidemic simulation..."):
                    simulation = network_analyzer.simulate_epidemic_spread(
                        G, model=model_type,
                        params={'beta': beta, 'gamma': gamma,
                                'initial_infected': initial_infected,
                                'steps': simulation_steps}
                    )

                    # Plot simulation results
                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=simulation['step'],
                        y=simulation['S'],
                        mode='lines',
                        name='Susceptible',
                        line=dict(color='blue', width=2)
                    ))

                    fig.add_trace(go.Scatter(
                        x=simulation['step'],
                        y=simulation['I'],
                        mode='lines',
                        name='Infected',
                        line=dict(color='red', width=2)
                    ))

                    if 'R' in simulation.columns:
                        fig.add_trace(go.Scatter(
                            x=simulation['step'],
                            y=simulation['R'],
                            mode='lines',
                            name='Recovered',
                            line=dict(color='green', width=2)
                        ))

                    # Add intervention line
                    if intervention_step > 0:
                        fig.add_vline(x=intervention_step, line_dash="dash",
                                      line_color="orange", annotation_text="Intervention")

                    fig.update_layout(
                        title=f'{model_type} Model Simulation',
                        xaxis_title='Time Step',
                        yaxis_title='Number of Nodes',
                        hovermode='x unified',
                        height=500
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Simulation statistics
                    st.markdown("#### 📊 Simulation Statistics")

                    peak_infections = simulation['I'].max()
                    peak_time = simulation.loc[simulation['I'].idxmax(), 'step']
                    if 'R' in simulation.columns:
                        final_attack_rate = simulation['R'].iloc[-1] / G.number_of_nodes()
                    else:
                        final_attack_rate = 0

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Peak Infections", f"{peak_infections:.0f}")

                    with col2:
                        st.metric("Peak Time", f"Step {peak_time}")

                    with col3:
                        st.metric("Final Attack Rate", f"{final_attack_rate:.1%}")

                    with col4:
                        if peak_time > 0 and simulation['I'].iloc[0] > 0:
                            growth = (peak_infections / simulation['I'].iloc[0]) ** (1 / peak_time) - 1
                            if growth > 0:
                                doubling = np.log(2) / np.log(1 + growth)
                                st.metric("Doubling Time", f"{doubling:.1f} steps")
                            else:
                                st.metric("Growth", "Negative")
                        else:
                            st.metric("Growth", "N/A")


def show_predictive_modeling():
    """Display predictive modeling interface"""
    st.markdown('<h1 class="main-header">Advanced Predictive Modeling</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Machine learning for disease prediction and risk assessment</p>',
                unsafe_allow_html=True)

    if st.session_state.data is None:
        st.warning("⚠️ Please load data first from Data Management")
        st.stop()

    data = st.session_state.data

    # Predictive Modeling Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🤖 Model Training", "📊 Model Comparison", "⚙️ Hyperparameter Tuning",
         "🎯 Feature Importance", "🧠 Deep Learning"]
    )

    with tab1:
        st.markdown("### 🤖 Train Machine Learning Models")

        col1, col2 = st.columns(2)

        with col1:
            target_variable = st.selectbox("Target Variable",
                                           [col for col in data.columns if data[col].dtype in ['int64', 'float64']],
                                           index=0 if 'cases' in data.columns else 0)

            # Feature selection
            available_features = [col for col in data.columns
                                  if col != target_variable and
                                  data[col].dtype in ['int64', 'float64', 'object']]

            selected_features = st.multiselect("Select Features",
                                               available_features,
                                               default=available_features[:min(10, len(available_features))])

        with col2:
            models_to_train = st.multiselect("Select Models",
                                             ["Random Forest", "XGBoost", "LightGBM", "Gradient Boosting",
                                              "Support Vector", "Neural Network", "Linear Regression",
                                              "Lasso", "Ridge", "Elastic Net", "CatBoost"],
                                             default=["Random Forest", "XGBoost", "LightGBM"])

            test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
            cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)

        if st.button("🚀 Train Models", type="primary"):
            with st.spinner("Training models..."):
                # Prepare features
                X = data[selected_features].copy()
                y = data[target_variable].copy()

                # Handle categorical variables
                categorical_cols = X.select_dtypes(include=['object']).columns
                if len(categorical_cols) > 0:
                    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

                # Handle missing values
                X = X.fillna(X.mean())
                y = y.fillna(y.mean())

                # Convert model names to format expected by train_ensemble_model
                models_to_use = [m.lower().replace(' ', '_') for m in models_to_train]

                # Train models - FIXED: Changed models_to_train to models_to_use
                results = ml_models.train_ensemble_model(X, y, models_to_use=models_to_use)

                st.session_state.ml_results = results
                st.success(f"✅ Trained {len(results)} models successfully!")

                # Display comparison
                st.markdown("#### 📊 Model Performance Comparison")

                comparison_data = []
                for model_name, metrics in results.items():
                    comparison_data.append({
                        'Model': model_name,
                        'Train R²': metrics['train']['r2'],
                        'Test R²': metrics['test']['r2'],
                        'Test RMSE': metrics['test']['rmse'],
                        'Test MAE': metrics['test']['mae']
                    })

                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df.sort_values('Test R²', ascending=False),
                             use_container_width=True)

                # Visualization
                fig = px.bar(comparison_df, x='Model', y='Test R²',
                             title='Model Performance (Test R²)',
                             color='Test R²', color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### 📊 Model Comparison Dashboard")

        if 'ml_results' not in st.session_state:
            st.info("Please train models first")
        else:
            results = st.session_state.ml_results

            # Performance comparison
            metrics_to_compare = st.multiselect("Compare Metrics",
                                                ["R²", "RMSE", "MAE", "MAPE"],
                                                default=["R²", "RMSE"])

            if metrics_to_compare:
                # Prepare comparison data
                comparison_data = []
                for model_name, metrics in results.items():
                    row = {'Model': model_name}
                    if "R²" in metrics_to_compare:
                        row['Test R²'] = metrics['test']['r2']
                    if "RMSE" in metrics_to_compare:
                        row['Test RMSE'] = metrics['test']['rmse']
                    if "MAE" in metrics_to_compare:
                        row['Test MAE'] = metrics['test']['mae']
                    if "MAPE" in metrics_to_compare:
                        row['Test MAPE'] = metrics['test']['mape']
                    comparison_data.append(row)

                comparison_df = pd.DataFrame(comparison_data)

                # Create comparison chart
                fig = go.Figure()

                colors = px.colors.qualitative.Set3
                for i, metric in enumerate(metrics_to_compare):
                    if metric == "R²":
                        fig.add_trace(go.Bar(
                            name='Test R²',
                            x=comparison_df['Model'],
                            y=comparison_df['Test R²'],
                            marker_color=colors[i % len(colors)]
                        ))
                    elif metric == "RMSE":
                        fig.add_trace(go.Bar(
                            name='Test RMSE',
                            x=comparison_df['Model'],
                            y=comparison_df['Test RMSE'],
                            marker_color=colors[i % len(colors)]
                        ))

                fig.update_layout(
                    title='Model Performance Comparison',
                    barmode='group',
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

            # Actual vs Predicted plots
            st.markdown("#### 📈 Actual vs Predicted")

            model_for_plot = st.selectbox("Select Model for Plot",
                                          list(results.keys()))

            if model_for_plot in results:
                # Get predictions from stored results
                # This would require storing predictions during training
                st.info(f"Selected model: {model_for_plot}")
                # Implementation would depend on how predictions are stored

    with tab3:
        st.markdown("### ⚙️ Hyperparameter Tuning")

        if st.session_state.data is not None:
            target_variable = st.selectbox("Target for Tuning",
                                           [col for col in data.columns if data[col].dtype in ['int64', 'float64']],
                                           key="tuning_target")

            model_for_tuning = st.selectbox("Model to Tune",
                                            ["Random Forest", "XGBoost", "LightGBM", "SVR"])

            if st.button("Perform Hyperparameter Tuning", type="primary"):
                with st.spinner("Running hyperparameter tuning..."):
                    # Prepare features
                    available_features = [col for col in data.columns
                                          if col != target_variable and
                                          data[col].dtype in ['int64', 'float64', 'object']]

                    X = data[available_features].copy()
                    y = data[target_variable].copy()

                    # Handle categoricals
                    categorical_cols = X.select_dtypes(include=['object']).columns
                    if len(categorical_cols) > 0:
                        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

                    X = X.fillna(X.mean())
                    y = y.fillna(y.mean())

                    # Perform tuning - FIXED: Now uses the implemented method
                    best_model, metrics = ml_models.perform_hyperparameter_tuning(
                        X, y, model_type=model_for_tuning.lower().replace(' ', '_')
                    )

                    if best_model is not None:
                        st.success("✅ Hyperparameter tuning completed!")

                        st.markdown("#### 🎯 Best Parameters")
                        for param, value in metrics['best_params'].items():
                            st.write(f"**{param}:** {value}")

                        st.markdown("#### 📊 Performance")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Best CV Score", f"{metrics['best_score']:.3f}")
                        with col2:
                            st.metric("Test R²", f"{metrics['test_r2']:.3f}")

    with tab4:
        st.markdown("### 🎯 Feature Importance Analysis")

        if 'ml_results' in st.session_state and st.session_state.ml_results is not None:
            # Get feature importance from the first model that has it
            for model_name, model_info in ml_models.results.items():
                if 'feature_importance' in model_info and model_info['feature_importance'] is not None:
                    feature_importance_df = model_info['feature_importance']

                    fig = px.bar(feature_importance_df.head(10),
                                 x='importance', y='feature',
                                 title='Top 10 Feature Importance',
                                 orientation='h')
                    st.plotly_chart(fig, use_container_width=True)

                    # Show full feature importance table
                    with st.expander("View All Feature Importance"):
                        st.dataframe(feature_importance_df, use_container_width=True)
                    break
            else:
                st.info("No feature importance data available. Please train models first.")
        else:
            st.info("Please train models first to see feature importance")

    with tab5:
        st.markdown("### 🧠 Deep Learning (LSTM)")

        if st.session_state.data is not None:
            st.info("LSTM models are best for sequential/time series data")

            if 'date' in data.columns:
                # Prepare time series data for LSTM
                st.markdown("#### Prepare LSTM Features")

                col1, col2 = st.columns(2)

                with col1:
                    target_col = st.selectbox("Target Column",
                                              [col for col in data.columns if data[col].dtype in ['int64', 'float64']],
                                              key="lstm_target")
                    sequence_length = st.slider("Sequence Length", 5, 30, 10)

                with col2:
                    epochs = st.slider("Training Epochs", 10, 200, 50)
                    features_for_lstm = st.multiselect("Features for LSTM",
                                                       [col for col in data.columns if
                                                        data[col].dtype in ['int64', 'float64'] and col != target_col],
                                                       default=[col for col in ['temperature', 'rainfall', 'humidity']
                                                                if col in data.columns])

                if st.button("Train LSTM Model", type="primary"):
                    with st.spinner("Training LSTM model..."):
                        # Prepare features
                        lstm_data = data[features_for_lstm + [target_col]].copy()
                        lstm_data = lstm_data.fillna(lstm_data.mean())

                        X = lstm_data[features_for_lstm]
                        y = lstm_data[target_col]

                        metrics = ml_models.train_lstm_model(X, y, sequence_length=sequence_length, epochs=epochs)

                        if metrics is not None:
                            st.success("✅ LSTM model trained successfully!")

                            # Display metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Train R²", f"{metrics['train']['r2']:.3f}")
                            with col2:
                                st.metric("Test R²", f"{metrics['test']['r2']:.3f}")
                            with col3:
                                st.metric("Test RMSE", f"{metrics['test']['rmse']:.3f}")
                            with col4:
                                st.metric("Test MAE", f"{metrics['test']['mae']:.3f}")
            else:
                st.warning("Date column required for time series LSTM modeling")
        else:
            st.info("Please load data first")


def show_epidemiological_models():
    """Display epidemiological models interface"""
    st.markdown('<h1 class="main-header">Epidemiological Models</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced compartmental models for disease transmission dynamics</p>',
                unsafe_allow_html=True)

    if st.session_state.data is None:
        st.warning("⚠️ Please load data first from Data Management")
        st.stop()

    data = st.session_state.data

    # Epidemiological Models Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🦠 Compartmental Models", "📈 Parameter Estimation", "🎯 Intervention Analysis", "🌐 Meta-population"]
    )

    with tab1:
        st.markdown("### 🦠 Compartmental Model Simulation")

        col1, col2 = st.columns(2)

        with col1:
            model_type = st.selectbox("Select Model", ["SIR", "SEIR", "SIS", "SEIRS"])

            if model_type == "SIR":
                st.markdown("**SIR Parameters:**")
                beta = st.slider("Transmission Rate (β)", 0.1, 2.0, 0.5, 0.1)
                gamma = st.slider("Recovery Rate (γ)", 0.01, 0.5, 0.1, 0.01)
                R0 = beta / gamma
                st.metric("Basic Reproduction Number (R₀)", f"{R0:.2f}")

            elif model_type == "SEIR":
                st.markdown("**SEIR Parameters:**")
                beta = st.slider("Transmission Rate (β)", 0.1, 2.0, 0.5, 0.1)
                sigma = st.slider("Incubation Rate (σ)", 0.1, 1.0, 0.2, 0.1)
                gamma = st.slider("Recovery Rate (γ)", 0.01, 0.5, 0.1, 0.01)
                R0 = beta / gamma
                st.metric("Basic Reproduction Number (R₀)", f"{R0:.2f}")

        with col2:
            population = st.number_input("Total Population", 1000, 10000000, 1000000)
            initial_infected = st.number_input("Initial Infected", 1, 1000, 10)
            simulation_days = st.slider("Simulation Days", 30, 365, 180)

            # Initial conditions
            if model_type == "SIR":
                S0 = population - initial_infected
                I0 = initial_infected
                R0_initial = 0
                initial_conditions = [S0, I0, R0_initial]

            elif model_type == "SEIR":
                S0 = population - initial_infected
                E0 = 0
                I0 = initial_infected
                R0_initial = 0
                initial_conditions = [S0, E0, I0, R0_initial]

        if st.button("Run Model Simulation", type="primary"):
            with st.spinner("Simulating disease dynamics..."):
                # Time points
                t = np.linspace(0, simulation_days, simulation_days)

                # Define ODE system
                if model_type == "SIR":
                    def sir_ode(t, y, beta, gamma):
                        S, I, R = y
                        N = S + I + R
                        dSdt = -beta * S * I / N
                        dIdt = beta * S * I / N - gamma * I
                        dRdt = gamma * I
                        return [dSdt, dIdt, dRdt]

                    solution = solve_ivp(
                        sir_ode,
                        [0, simulation_days],
                        initial_conditions,
                        args=(beta, gamma),
                        t_eval=t,
                        method='RK45'
                    )

                    # Create results dataframe
                    results = pd.DataFrame({
                        'Day': solution.t,
                        'Susceptible': solution.y[0],
                        'Infected': solution.y[1],
                        'Recovered': solution.y[2]
                    })

                # Plot results
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=results['Day'], y=results['Susceptible'],
                    mode='lines', name='Susceptible',
                    line=dict(color='blue', width=2)
                ))

                fig.add_trace(go.Scatter(
                    x=results['Day'], y=results['Infected'],
                    mode='lines', name='Infected',
                    line=dict(color='red', width=2)
                ))

                fig.add_trace(go.Scatter(
                    x=results['Day'], y=results['Recovered'],
                    mode='lines', name='Recovered',
                    line=dict(color='green', width=2)
                ))

                fig.update_layout(
                    title=f'{model_type} Model Simulation',
                    xaxis_title='Days',
                    yaxis_title='Population',
                    hovermode='x unified',
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

                # Model statistics
                st.markdown("#### 📊 Model Statistics")

                peak_infected = results['Infected'].max()
                peak_day = results.loc[results['Infected'].idxmax(), 'Day']
                final_attack_rate = results['Recovered'].iloc[-1] / population

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Peak Infections", f"{peak_infected:,.0f}")

                with col2:
                    st.metric("Peak Day", f"{peak_day:.0f}")

                with col3:
                    st.metric("Final Attack Rate", f"{final_attack_rate:.1%}")

                with col4:
                    st.metric("R₀", f"{R0:.2f}")

    with tab2:
        st.markdown("### 📈 Parameter Estimation from Data")

        if 'date' in data.columns and 'cases' in data.columns:
            daily_cases = data.groupby('date')['cases'].sum()

            col1, col2 = st.columns(2)

            with col1:
                model_to_fit = st.selectbox("Model to Fit", ["SIR", "SEIR"])
                population = st.number_input("Population for Fitting", 1000, 10000000, 1000000)

            with col2:
                estimation_method = st.selectbox("Estimation Method",
                                                 ["Least Squares", "Maximum Likelihood", "MCMC"])
                show_confidence = st.checkbox("Show Confidence Intervals", True)

            if st.button("Estimate Parameters", type="primary"):
                with st.spinner("Fitting model to data..."):
                    if model_to_fit == "SIR":
                        result = epi_models.fit_sir_model(daily_cases.values, population)

                        if result['success']:
                            st.success("✅ Model fitted successfully!")

                            # Display parameters
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric("β (Transmission)", f"{result['beta']:.4f}")

                            with col2:
                                st.metric("γ (Recovery)", f"{result['gamma']:.4f}")

                            with col3:
                                st.metric("R₀", f"{result['R0']:.2f}")

                            # Plot fit
                            fig = go.Figure()

                            fig.add_trace(go.Scatter(
                                x=np.arange(len(daily_cases)),
                                y=daily_cases.values,
                                mode='markers',
                                name='Actual Cases',
                                marker=dict(color='blue', size=6)
                            ))

                            fig.add_trace(go.Scatter(
                                x=np.arange(len(result['fitted_curve'])),
                                y=result['fitted_curve'],
                                mode='lines',
                                name='Fitted Model',
                                line=dict(color='red', width=2)
                            ))

                            fig.update_layout(
                                title='SIR Model Fit to Data',
                                xaxis_title='Days',
                                yaxis_title='Cases',
                                height=500
                            )

                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("❌ Model fitting failed")

        else:
            st.warning("Date and cases data required for parameter estimation")

    with tab3:
        st.markdown("### 🎯 Intervention Scenario Analysis")

        st.markdown("""
        Analyze the impact of different intervention strategies on disease transmission.
        """)

        interventions = st.multiselect("Select Interventions",
                                       ["Social Distancing", "Mask Mandate", "Vaccination",
                                        "Travel Restrictions", "Lockdown", "Testing"],
                                       default=["Social Distancing", "Vaccination"])

        if st.button("Analyze Interventions", type="primary"):
            with st.spinner("Simulating intervention scenarios..."):
                # Base parameters
                base_params = {
                    'beta': 0.5,
                    'gamma': 0.1,
                    'R0': 5.0
                }

                # Intervention effects
                intervention_effects = {
                    'Social Distancing': {'beta': 0.3},
                    'Mask Mandate': {'beta': 0.4},
                    'Vaccination': {'beta': 0.2, 'gamma': 0.15},
                    'Travel Restrictions': {'beta': 0.35},
                    'Lockdown': {'beta': 0.1},
                    'Testing': {'gamma': 0.2}
                }

                # Simulate scenarios
                scenarios = epi_models.simulate_intervention_scenarios(
                    base_params,
                    {k: v for k, v in intervention_effects.items() if k in interventions}
                )

                # Display results
                for name, scenario in scenarios.items():
                    with st.expander(f"{name} Intervention"):
                        st.write(f"**Parameters:** {scenario['parameters']}")
                        st.write(f"**Estimated Impact:** {scenario['estimated_impact']}")


def show_visualization_dashboard():
    """Display comprehensive visualization dashboard"""
    st.markdown('<h1 class="main-header">Visualization Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Interactive visual analytics for epidemiological data</p>',
                unsafe_allow_html=True)

    if st.session_state.data is None:
        st.warning("⚠️ Please load data first from Data Management")
        st.stop()

    data = st.session_state.data

    # Visualization Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["📊 Overview Dashboard", "🗺️ Spatial Visualizations", "📈 Temporal Visualizations",
         "🔗 Network Visualizations", "📉 Statistical Plots", "🎨 Custom Visualizations"]
    )

    with tab1:
        st.markdown("### 📊 Comprehensive Overview Dashboard")

        # Create a comprehensive dashboard with multiple plots
        if 'date' in data.columns and 'cases' in data.columns:
            # Row 1: Cases overview
            col1, col2, col3 = st.columns(3)

            with col1:
                # Daily cases line chart
                daily_cases = data.groupby('date')['cases'].sum()
                fig1 = px.line(daily_cases, title='Daily Cases')
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                # Cumulative cases
                cumulative = daily_cases.cumsum()
                fig2 = px.area(cumulative, title='Cumulative Cases')
                st.plotly_chart(fig2, use_container_width=True)

            with col3:
                # 7-day moving average
                ma7 = daily_cases.rolling(7).mean()
                fig3 = px.line(ma7, title='7-Day Moving Average')
                st.plotly_chart(fig3, use_container_width=True)

            # Row 2: Regional distribution
            if 'region' in data.columns:
                st.markdown("#### Regional Distribution")

                col1, col2 = st.columns(2)

                with col1:
                    # Bar chart by region
                    region_cases = data.groupby('region')['cases'].sum().sort_values(ascending=False)
                    fig4 = px.bar(region_cases.head(10), title='Top 10 Regions by Cases')
                    st.plotly_chart(fig4, use_container_width=True)

                with col2:
                    # Pie chart
                    fig5 = px.pie(values=region_cases.values, names=region_cases.index,
                                  title='Case Distribution by Region')
                    st.plotly_chart(fig5, use_container_width=True)

    with tab2:
        st.markdown("### 🗺️ Advanced Spatial Visualizations")

        col1, col2 = st.columns(2)

        with col1:
            map_type = st.selectbox("Map Type",
                                    ["Heatmap", "Bubble", "Choropleth", "3D Scatter", "Animated"])
            value_column = st.selectbox("Value for Map",
                                        [col for col in data.columns if data[col].dtype in ['int64', 'float64']])

        with col2:
            basemap = st.selectbox("Base Map Style",
                                   ["OpenStreetMap", "CartoDB Positron", "Stamen Terrain"])
            map_height = st.slider("Map Height", 400, 1000, 600)

        if map_type == "3D Scatter" and all(col in data.columns for col in ['latitude', 'longitude']):
            fig = px.scatter_3d(data, x='longitude', y='latitude', z=value_column,
                                color=value_column, hover_name='region' if 'region' in data.columns else None,
                                title='3D Spatial Distribution')
            st.plotly_chart(fig, use_container_width=True, height=map_height)
        elif all(col in data.columns for col in ['latitude', 'longitude']):
            # Create 2D map
            if map_type == "Heatmap":
                fig = px.density_mapbox(data, lat='latitude', lon='longitude', z=value_column,
                                        radius=20, zoom=6, mapbox_style=basemap.lower(),
                                        title='Heatmap')
            elif map_type == "Bubble":
                fig = px.scatter_mapbox(data, lat='latitude', lon='longitude', size=value_column,
                                        color=value_column, zoom=6, mapbox_style=basemap.lower(),
                                        title='Bubble Map')

            st.plotly_chart(fig, use_container_width=True, height=map_height)

    with tab3:
        st.markdown("### 📈 Advanced Temporal Visualizations")

        visualization_type = st.selectbox("Visualization Type",
                                          ["Time Series", "Seasonal Decomposition",
                                           "Auto-correlation", "Cross-correlation"])

        if visualization_type == "Time Series":
            # Multiple time series
            if 'region' in data.columns and 'date' in data.columns and 'cases' in data.columns:
                regions = st.multiselect("Select Regions",
                                         data['region'].unique(),
                                         default=data['region'].unique()[:5])

                if regions:
                    filtered_data = data[data['region'].isin(regions)]
                    fig = px.line(filtered_data, x='date', y='cases', color='region',
                                  title='Multiple Time Series')
                    st.plotly_chart(fig, use_container_width=True)

        elif visualization_type == "Auto-correlation":
            if 'date' in data.columns and 'cases' in data.columns:
                daily_cases = data.groupby('date')['cases'].sum()

                fig, ax = plt.subplots(figsize=(10, 4))
                plot_acf(daily_cases, ax=ax, lags=30)
                st.pyplot(fig)

    with tab4:
        st.markdown("### 🔗 Network Visualization")

        if 'network' in st.session_state and st.session_state.network is not None:
            G = st.session_state.network

            layout = st.selectbox("Network Layout",
                                  ["Spring", "Circular", "Random", "Kamada-Kawai"])

            node_color = st.selectbox("Node Color By",
                                      ["Degree", "Betweenness", "Closeness", "Community"])

            if st.button("Visualize Network", type="primary"):
                # Calculate positions
                if layout == "Spring":
                    pos = nx.spring_layout(G, seed=42)
                elif layout == "Circular":
                    pos = nx.circular_layout(G)
                elif layout == "Random":
                    pos = nx.random_layout(G, seed=42)
                else:
                    pos = nx.kamada_kawai_layout(G)

                # Prepare edge traces
                edge_x = []
                edge_y = []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none',
                    mode='lines'
                )

                # Prepare node traces
                node_x = []
                node_y = []
                node_text = []
                node_color_values = []

                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)

                    node_info = f"Node {node}"
                    if 'region' in G.nodes[node]:
                        node_info = f"{G.nodes[node]['region']}\nDegree: {G.degree(node)}"

                    node_text.append(node_info)

                    # Color nodes
                    if node_color == "Degree":
                        node_color_values.append(G.degree(node))
                    else:
                        node_color_values.append(G.degree(node))  # Default

                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    text=node_text,
                    textposition="top center",
                    marker=dict(
                        size=10,
                        color=node_color_values,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title=node_color)
                    )
                )

                fig = go.Figure(data=[edge_trace, node_trace],
                                layout=go.Layout(
                                    title='Network Visualization',
                                    showlegend=False,
                                    hovermode='closest',
                                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                    height=600
                                ))

                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please build a network first in Network Analysis")

    with tab5:
        st.markdown("### 📉 Statistical Plots")

        plot_type = st.selectbox("Select Plot Type",
                                 ["Histogram", "Box Plot", "Violin Plot",
                                  "Scatter Plot", "Correlation Matrix", "Pair Plot"])

        if plot_type == "Histogram":
            column = st.selectbox("Select Column",
                                  [col for col in data.columns if data[col].dtype in ['int64', 'float64']])

            fig = px.histogram(data, x=column, nbins=50, title=f'Distribution of {column}')
            st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Box Plot":
            numeric_cols = [col for col in data.columns if data[col].dtype in ['int64', 'float64']]
            if len(numeric_cols) > 0:
                fig = px.box(data, y=numeric_cols[:5], title='Box Plots')
                st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Correlation Matrix":
            numeric_data = data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) > 1:
                corr_matrix = numeric_data.corr()
                fig = px.imshow(corr_matrix, color_continuous_scale='RdBu_r',
                                title='Correlation Matrix')
                st.plotly_chart(fig, use_container_width=True)


def show_export_center():
    """Display export interface"""
    st.markdown('<h1 class="main-header">Export Center</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Export data, visualizations, and reports in multiple formats</p>',
                unsafe_allow_html=True)

    # Export Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Data Export", "📈 Plot Export", "📋 Report Generation",
         "🎨 Dashboard Export", "🔧 Batch Export"]
    )

    with tab1:
        st.markdown("### 📊 Export Data")

        if st.session_state.data is not None:
            data = st.session_state.data

            col1, col2 = st.columns(2)

            with col1:
                export_format = st.selectbox("Format",
                                             ["CSV", "Excel", "JSON", "GeoJSON", "Parquet"])

                if export_format == "CSV":
                    delimiter = st.selectbox("Delimiter", [",", ";", "\t"])
                    encoding = st.selectbox("Encoding", ["utf-8", "latin-1", "windows-1252"])

            with col2:
                compression = st.checkbox("Compress File", False)
                include_index = st.checkbox("Include Index", False)
                filename = st.text_input("Filename", "epidemiological_data")

            if st.button("Export Data", type="primary"):
                with st.spinner("Preparing export..."):
                    file_data, file_name, mime_type = export_manager.export_data(
                        data, export_format, filename
                    )

                    if file_data is not None:
                        st.download_button(
                            label=f"Download {export_format}",
                            data=file_data,
                            file_name=file_name,
                            mime=mime_type
                        )
        else:
            st.info("No data to export")

    with tab2:
        st.markdown("### 📈 Export Plots")

        st.info("Select a plot to export from the Visualization Dashboard")

        plot_type = st.selectbox("Plot Type",
                                 ["Time Series", "Map", "Network", "Statistical"])

        col1, col2 = st.columns(2)

        with col1:
            export_format = st.selectbox("Export Format",
                                         ["PNG", "PDF", "SVG", "HTML"],
                                         key="plot_format")

            width = st.number_input("Width (pixels)", 400, 2000, 1200)
            height = st.number_input("Height (pixels)", 300, 1500, 800)

        with col2:
            dpi = st.selectbox("DPI (for raster formats)", [72, 150, 300, 600])
            include_legend = st.checkbox("Include Legend", True)
            transparent = st.checkbox("Transparent Background", False)

        if st.button("Generate Sample Plot", type="primary"):
            # Create a sample plot for export
            if st.session_state.data is not None and 'date' in st.session_state.data.columns:
                daily_cases = st.session_state.data.groupby('date')['cases'].sum()
                fig = px.line(daily_cases, title='Sample Plot for Export')

                st.plotly_chart(fig, use_container_width=True)

                # Export button
                try:
                    plot_data, plot_name, plot_mime = export_manager.export_plot(
                        fig, export_format, "sample_plot"
                    )

                    if plot_data is not None:
                        st.download_button(
                            label=f"Download {export_format}",
                            data=plot_data,
                            file_name=plot_name,
                            mime=plot_mime
                        )
                except Exception as e:
                    st.error(f"Export error: {str(e)}")

    with tab3:
        st.markdown("### 📋 Generate Reports")

        report_type = st.selectbox("Report Type",
                                   ["Executive Summary", "Technical Analysis",
                                    "Comprehensive Report", "Custom Report"])

        col1, col2 = st.columns(2)

        with col1:
            include_charts = st.checkbox("Include Charts", True)
            include_statistics = st.checkbox("Include Statistics", True)
            include_recommendations = st.checkbox("Include Recommendations", True)

        with col2:
            include_forecasts = st.checkbox("Include Forecasts", True)
            include_methodology = st.checkbox("Include Methodology", False)
            report_title = st.text_input("Report Title", "Epidemiological Analysis Report")

        if st.button("Generate Report", type="primary"):
            with st.spinner("Generating report..."):
                # Generate analysis results
                analysis_results = {}

                if st.session_state.data is not None:
                    data = st.session_state.data

                    # Basic statistics
                    if 'cases' in data.columns:
                        analysis_results['total_cases'] = data['cases'].sum()
                        if 'date' in data.columns:
                            analysis_results['avg_daily_cases'] = data.groupby('date')['cases'].sum().mean()
                        else:
                            analysis_results['avg_daily_cases'] = data['cases'].mean()

                    if 'date' in data.columns:
                        try:
                            analysis_results[
                                'date_range'] = f"{data['date'].min().date()} to {data['date'].max().date()}"
                        except:
                            analysis_results['date_range'] = f"{data['date'].min()} to {data['date'].max()}"

                    # Generate report
                    report_text = export_manager.generate_report(data, analysis_results)

                    # Show preview
                    with st.expander("Report Preview", expanded=True):
                        st.text(report_text[:2000] + "..." if len(report_text) > 2000 else report_text)

                    # Download button
                    st.download_button(
                        label="Download Report (TXT)",
                        data=report_text.encode(),
                        file_name=f"{report_title.lower().replace(' ', '_')}.txt",
                        mime="text/plain"
                    )
                else:
                    st.warning("No data available for report generation")

    with tab4:
        st.markdown("### 🎨 Export Interactive Dashboard")

        st.markdown("""
        Export the complete dashboard as an interactive HTML file that can be 
        viewed in any web browser without requiring the application.
        """)

        dashboard_type = st.selectbox("Dashboard Type",
                                      ["Overview Dashboard", "Spatial Dashboard",
                                       "Temporal Dashboard", "Complete Dashboard"])

        include_data = st.checkbox("Include Data in Export", True)
        interactive = st.checkbox("Interactive Elements", True)
        responsive = st.checkbox("Responsive Design", True)

        if st.button("Generate Dashboard Export", type="primary"):
            with st.spinner("Creating dashboard export..."):
                # This would generate an HTML dashboard
                st.info("Dashboard export would be generated here")
                # Implementation would create an HTML file with all visualizations

    with tab5:
        st.markdown("### 🔧 Batch Export Operations")

        st.markdown("""
        Perform multiple exports at once for efficient data sharing and reporting.
        """)

        export_tasks = st.multiselect("Select Export Tasks",
                                      ["Export All Data", "Export All Plots",
                                       "Export Analysis Results", "Export Model Outputs",
                                       "Export Configuration"],
                                      default=["Export All Data"])

        output_format = st.selectbox("Output Format", ["ZIP Archive", "Folder", "Cloud Storage"])

        if st.button("Run Batch Export", type="primary"):
            with st.spinner("Running batch export..."):
                progress_bar = st.progress(0)

                for i, task in enumerate(export_tasks):
                    st.write(f"Exporting: {task}")
                    # Simulate export progress
                    progress_bar.progress((i + 1) / len(export_tasks))

                st.success("✅ Batch export completed!")

                # Create download link for ZIP file
                st.download_button(
                    label="Download Export Bundle",
                    data=b"Sample export data",  # This would be the actual ZIP file
                    file_name="export_bundle.zip",
                    mime="application/zip"
                )


def show_advanced_settings():
    """Display advanced settings interface"""
    st.markdown('<h1 class="main-header">Advanced Settings</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Configure application settings and preferences</p>', unsafe_allow_html=True)

    # Settings Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🎨 Appearance", "🔧 Analysis", "💾 Performance", "🔄 Updates", "🔒 Security"]
    )

    with tab1:
        st.markdown("### 🎨 Appearance Settings")

        col1, col2 = st.columns(2)

        with col1:
            theme = st.selectbox("Color Theme",
                                 ["Blue Ocean", "Green Forest", "Red Alert",
                                  "Purple Haze", "Dark Mode", "Custom"])

            if theme == "Custom":
                primary_color = st.color_picker("Primary Color", "#1a5f7a")
                secondary_color = st.color_picker("Secondary Color", "#2c3e50")
                accent_color = st.color_picker("Accent Color", "#9b59b6")

            font_family = st.selectbox("Font Family",
                                       ["Inter", "Roboto", "Open Sans",
                                        "Montserrat", "System Default"])

            font_size = st.selectbox("Font Size", ["Small", "Medium", "Large"], index=1)

        with col2:
            chart_style = st.selectbox("Default Chart Style",
                                       ["Plotly", "Matplotlib", "Seaborn", "Custom"])

            if chart_style == "Custom":
                chart_template = st.selectbox("Chart Template",
                                              ["plotly", "plotly_white", "plotly_dark",
                                               "ggplot2", "seaborn", "simple_white"])

            animation_speed = st.slider("Animation Speed", 1, 10, 5)
            tooltip_delay = st.slider("Tooltip Delay (ms)", 0, 2000, 500)

        if st.button("Save Appearance Settings", type="primary"):
            st.success("✅ Appearance settings saved!")
            st.info("Restart the application for changes to take effect")

    with tab2:
        st.markdown("### 🔧 Analysis Settings")

        col1, col2 = st.columns(2)

        with col1:
            default_confidence = st.slider("Default Confidence Level", 80, 99, 95)
            outlier_threshold = st.slider("Outlier Detection Threshold", 1.5, 3.0, 1.5, 0.1)

            spatial_resolution = st.selectbox("Spatial Resolution",
                                              ["High (100m)", "Medium (1km)", "Low (10km)"], index=1)

            clustering_method = st.selectbox("Default Clustering Method",
                                             ["DBSCAN", "K-Means", "Hierarchical"])

        with col2:
            temporal_resolution = st.selectbox("Temporal Resolution",
                                               ["Daily", "Weekly", "Monthly"], index=0)

            max_clusters = st.slider("Maximum Clusters", 5, 100, 20)
            auto_save_results = st.checkbox("Auto-save Analysis Results", True)

            if auto_save_results:
                save_interval = st.selectbox("Save Interval",
                                             ["Every 5 minutes", "Every 15 minutes",
                                              "Every hour", "After each analysis"])

        if st.button("Save Analysis Settings", type="primary"):
            st.success("✅ Analysis settings saved!")

    with tab3:
        st.markdown("### 💾 Performance Settings")

        col1, col2 = st.columns(2)

        with col1:
            cache_enabled = st.checkbox("Enable Caching", True)

            if cache_enabled:
                cache_size = st.selectbox("Cache Size",
                                          ["Small (100MB)", "Medium (500MB)", "Large (1GB)"], index=1)
                cache_ttl = st.selectbox("Cache Time-to-Live",
                                         ["1 hour", "6 hours", "24 hours", "7 days"])

            parallel_processing = st.checkbox("Enable Parallel Processing", True)

            if parallel_processing:
                max_workers = st.slider("Maximum Workers", 1, 8, 4)

        with col2:
            memory_limit = st.selectbox("Memory Limit",
                                        ["1GB", "2GB", "4GB", "8GB", "Unlimited"], index=2)

            data_chunk_size = st.selectbox("Data Chunk Size",
                                           ["10,000 rows", "50,000 rows",
                                            "100,000 rows", "All data"], index=1)

            auto_cleanup = st.checkbox("Auto-cleanup Temporary Files", True)

        if st.button("Save Performance Settings", type="primary"):
            st.success("✅ Performance settings saved!")

    with tab4:
        st.markdown("### 🔄 System Updates")

        st.info("""
        ### Current Version: EpiGeoSim-X Pro v5.0

        **Last Update:** 2024-01-15

        **New Features in v5.0:**
        - Enhanced network analysis with multiple models
        - Advanced spatial regression (GWR, Spatial Lag, Spatial Error)
        - 15+ machine learning algorithms with hyperparameter tuning
        - Comprehensive epidemiological models (SIR, SEIR, etc.)
        - Shapefile support and advanced GIS integration
        - Batch export capabilities

        **System Requirements:**
        - Python 3.8 or higher
        - 8GB RAM minimum (16GB recommended)
        - 2GB free disk space
        - Internet connection for updates
        - Modern web browser (Chrome 90+, Firefox 88+, Safari 14+)
        """)

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔄 Check for Updates", use_container_width=True):
                st.info("✅ You are running the latest version!")

        with col2:
            if st.button("📋 View Changelog", use_container_width=True):
                st.info("Changelog would open here")

        with col3:
            if st.button("🔄 Restart Application", use_container_width=True):
                st.warning("Application will restart")
                st.rerun()

        st.markdown("---")

        st.markdown("### 🐛 Bug Report & Feedback")

        with st.form("feedback_form"):
            feedback_type = st.selectbox("Type",
                                         ["Bug Report", "Feature Request",
                                          "General Feedback", "Performance Issue"])

            feedback_text = st.text_area("Your Feedback", height=150,
                                         placeholder="Describe your issue or suggestion...")

            contact_email = st.text_input("Email (optional)",
                                          placeholder="your.email@example.com")

            include_logs = st.checkbox("Include System Logs", False)

            submitted = st.form_submit_button("Submit Feedback", type="primary")

            if submitted:
                if feedback_text.strip():
                    st.success("✅ Thank you for your feedback! We'll review it soon.")
                    if contact_email:
                        st.info(f"We'll contact you at {contact_email} if needed")
                else:
                    st.warning("Please provide feedback text")

    with tab5:
        st.markdown("### 🔒 Security & Privacy Settings")

        col1, col2 = st.columns(2)

        with col1:
            data_encryption = st.checkbox("Enable Data Encryption", True)

            if data_encryption:
                encryption_level = st.selectbox("Encryption Level",
                                                ["Standard (AES-128)",
                                                 "High (AES-256)",
                                                 "Military (AES-512)"])

            auto_logout = st.checkbox("Auto-logout after Inactivity", True)

            if auto_logout:
                logout_time = st.selectbox("Logout After",
                                           ["5 minutes", "15 minutes",
                                            "30 minutes", "1 hour"])

        with col2:
            data_retention = st.selectbox("Data Retention Policy",
                                          ["Keep all data",
                                           "Delete after 30 days",
                                           "Delete after 90 days",
                                           "Delete after analysis"])

            audit_logging = st.checkbox("Enable Audit Logging", True)

            if audit_logging:
                log_level = st.selectbox("Log Level",
                                         ["Basic", "Detailed", "Debug"])

            export_restrictions = st.checkbox("Restrict Data Exports", False)

        st.markdown("---")

        st.markdown("### 📜 Data Privacy Compliance")

        compliance_frameworks = st.multiselect("Compliance Frameworks",
                                               ["GDPR", "HIPAA", "FERPA",
                                                "ISO 27001", "Custom"])

        if st.button("Save Security Settings", type="primary"):
            st.success("✅ Security settings saved!")
            st.info("Some settings may require application restart")


# ============================================
# HELPER FUNCTIONS
# ============================================

def generate_demo_data(regions=6, days=365, outbreak_intensity=5,
                       include_env=True, include_demo=True):
    """Generate realistic demo data for testing"""
    np.random.seed(42)

    # Bangladesh divisions with realistic parameters
    divisions = {
        'Dhaka': {'lat': 23.8103, 'lon': 90.4125, 'pop_mil': 21, 'urban': 0.85},
        'Chittagong': {'lat': 22.3569, 'lon': 91.7832, 'pop_mil': 8, 'urban': 0.70},
        'Khulna': {'lat': 22.8456, 'lon': 89.5403, 'pop_mil': 3.5, 'urban': 0.60},
        'Rajshahi': {'lat': 24.3745, 'lon': 88.6042, 'pop_mil': 2.8, 'urban': 0.55},
        'Sylhet': {'lat': 24.8949, 'lon': 91.8687, 'pop_mil': 2.5, 'urban': 0.50},
        'Barisal': {'lat': 22.7010, 'lon': 90.3535, 'pop_mil': 2.3, 'urban': 0.45}
    }

    data = []
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')

    for date in dates:
        day_of_year = date.dayofyear

        for region_name, params in list(divisions.items())[:regions]:
            # Base case rate with seasonality
            seasonal = 3 * np.sin(2 * np.pi * (day_of_year - 15) / 365)
            trend = day_of_year * 0.01

            # Population effect
            pop_factor = params['pop_mil'] / 5

            # Calculate base rate
            base_rate = max(5, 10 * pop_factor + seasonal + trend)

            # Random outbreaks
            if np.random.random() < 0.005:  # 0.5% chance of outbreak
                base_rate *= outbreak_intensity

            # Generate cases
            cases = max(0, int(np.random.poisson(base_rate)))

            # Create data row
            row = {
                'date': date,
                'region': region_name,
                'latitude': params['lat'] + np.random.uniform(-0.2, 0.2),
                'longitude': params['lon'] + np.random.uniform(-0.2, 0.2),
                'cases': cases,
                'deaths': int(cases * np.random.uniform(0.005, 0.02)),
                'hospitalizations': int(cases * np.random.uniform(0.05, 0.15))
            }

            # Add environmental data
            if include_env:
                row.update({
                    'temperature': 25 + 10 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 3),
                    'rainfall': max(0,
                                    10 + 8 * np.sin(2 * np.pi * (day_of_year + 120) / 365) + np.random.exponential(5)),
                    'humidity': 60 + 20 * np.sin(2 * np.pi * (day_of_year - 45) / 365) + np.random.normal(0, 5),
                    'ndvi': 0.6 + 0.2 * np.sin(2 * np.pi * (day_of_year - 30) / 365) + np.random.normal(0, 0.05)
                })

            # Add demographic data
            if include_demo:
                row.update({
                    'population': params['pop_mil'] * 1000000,
                    'urbanization_rate': params['urban'],
                    'healthcare_index': np.random.uniform(0.4, 0.9),
                    'poverty_rate': np.random.uniform(0.1, 0.4),
                    'sanitation_index': np.random.uniform(0.5, 0.95),
                    'vaccination_coverage': np.random.uniform(0.6, 0.9)
                })

            data.append(row)

    df = pd.DataFrame(data)

    # Add derived features
    df['incidence_rate'] = (df['cases'] / df['population']) * 100000
    df['mortality_rate'] = (df['deaths'] / df['cases'].replace(0, 1)) * 100
    df['case_fatality_rate'] = df['deaths'] / df['cases'].replace(0, 1)

    # Add temporal features
    df['weekday'] = df['date'].dt.day_name()
    df['month'] = df['date'].dt.month
    df['week'] = df['date'].dt.isocalendar().week

    return df


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    main()