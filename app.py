"""
Customer Lifetime Value & Segmentation Engine  —  Production Edition
=====================================================================
NextGen Institute · ML Final Project (Project #02)
Dataset: UCI Online Retail II (~1M transactions, 2009-2011)

Key Capabilities
----------------
1. 📊 Overview         — KPIs, segment distributions, dynamic RFM explorer, correlation heatmap
2. 🔍 Customer Lookup  — Per-customer card with percentile, waterfall breakdown, spend history
3. 🔬 Train Your Model — Live in-app K-Means & CatBoost training with interactive hyperparams & metrics
4. 🎯 What-If Forecast — Real-time slider simulator, gauge chart vs baseline, RFM radar comparison
5. 🛡️ Smart Data Inspector & Auto-Predictor — Upload ANY dataset (clean or dirty, raw transactions or RFM).
   Automatically detects dtypes, syntax errors, missing values, auto-repairs dirty data, maps column synonyms,
   and predicts using any of the 7 saved models (or all simultaneously).
6. 📋 Methodology      — Complete pipeline explanation, metrics for all 7 models, limitations
"""
from __future__ import annotations

import io
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer LTV & Segmentation Engine",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT         = Path(__file__).resolve().parent
BUNDLE_PATH  = ROOT / "models" / "model_bundle.joblib"
PROFILE_PATH = ROOT / "data"   / "customer_profiles.csv"

SEG_COLORS = {
    "Champions":                    "#1a56db",
    "Active Customers":             "#16a34a",
    "At-Risk High-Value Customers": "#d97706",
    "Hibernating Customers":        "#6b7280",
}
SEG_COLOR_LIST = ["#1a56db", "#16a34a", "#d97706", "#6b7280", "#a855f7", "#ef4444"]

# ──────────────────────────────────────────────────────────────────────────────
# Residual & Candidate model classes
# ──────────────────────────────────────────────────────────────────────────────
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin


class ResidualCatBoostRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, iterations=300, depth=5, learning_rate=0.03,
                 l2_leaf_reg=5, random_seed=42):
        self.iterations    = iterations
        self.depth         = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg   = l2_leaf_reg
        self.random_seed   = random_seed

    def fit(self, X, y, sample_weight=None):
        self.model_ = CatBoostRegressor(
            iterations=self.iterations, depth=self.depth,
            learning_rate=self.learning_rate, l2_leaf_reg=self.l2_leaf_reg,
            loss_function="RMSE", verbose=False,
            random_seed=self.random_seed, thread_count=1,
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):        return self.model_.predict(X)
    def get_feature_importance(self): return self.model_.get_feature_importance()
    def save_model(self, path):  self.model_.save_model(str(path))


# ──────────────────────────────────────────────────────────────────────────────
# Multi-Model Scoring Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _baseline(X: pd.DataFrame) -> np.ndarray:
    if "R180_Spend" in X.columns:
        return X["R180_Spend"].to_numpy(dtype=float) / 2.0
    elif "Monetary" in X.columns:
        return X["Monetary"].to_numpy(dtype=float) * 0.25
    return np.zeros(len(X), dtype=float)


def score_customers(df: pd.DataFrame, bundle: dict, model_name: str = "CatBoost (Tuned)") -> pd.DataFrame:
    """Assign segment + action + spend forecast to every row using the selected model."""
    out = df.copy()
    rfm_cols     = bundle.get("rfm_columns", ["Recency", "Frequency", "Monetary"])
    feature_cols = bundle.get("feature_columns", [
        'Recency', 'Frequency', 'Monetary', 'UniqueProducts', 'AvgBasket',
        'TotalQuantity', 'TenureDays', 'R30_Spend', 'R60_Spend', 'R90_Spend',
        'R180_Spend', 'AvgOrderValue', 'SpendPerDay'
    ])

    # Check / synthesize missing columns if any
    for c in rfm_cols:
        if c not in out.columns:
            out[c] = 0.0
    for c in feature_cols:
        if c not in out.columns:
            if c == "AvgOrderValue":
                out[c] = (out["Monetary"] / out["Frequency"].clip(lower=1)).round(2)
            elif c == "R180_Spend":
                out[c] = out["Monetary"]
            else:
                out[c] = 0.0

    # 1. Segmentation via KMeans
    X_rfm = np.log1p(out[rfm_cols].clip(lower=0).astype(float))
    Z = bundle["cluster_scaler"].transform(X_rfm)
    out["Segment_ID"] = bundle["kmeans_model"].predict(Z).astype(int)
    out["Segment"]    = out["Segment_ID"].map(bundle["segment_names"]).fillna("Active Customers")

    # 2. Spend Forecast
    X_feat = out[feature_cols].astype(float)
    base   = _baseline(X_feat)

    # Determine which model to use
    all_models = bundle.get("all_models", {})
    if model_name in all_models:
        model = all_models[model_name]
    else:
        model = bundle.get("spend_regressor")

    raw_pred = model.predict(X_feat)
    
    if bundle.get("residual_target", False):
        uncalibrated = np.maximum(0.0, np.expm1(np.log1p(base) + raw_pred))
    else:
        uncalibrated = np.maximum(0.0, np.expm1(raw_pred))

    ml_adjusted = uncalibrated * bundle.get("calibration_factor", 1.0)
    blend_w = bundle.get("ml_blend_weight", 0.88) if model_name in ["CatBoost (Tuned)", "Ensemble Blend"] else 1.0
    blended = blend_w * ml_adjusted + (1.0 - blend_w) * base

    out["Baseline_90d_Spend"]     = base.round(2)
    out["ML_Adjusted_90d_Spend"]  = ml_adjusted.round(2)
    out["Predicted_Next90_Spend"] = np.maximum(0.0, blended).round(2)
    out["Recommended_Action"]     = out["Segment_ID"].map(bundle["segment_actions"]).fillna("Engage with personalized offers.")
    out["Model_Used"]             = model_name
    return out


def score_all_candidate_models(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Run all 5 candidate models simultaneously on the dataset for side-by-side comparison."""
    feature_cols = bundle.get("feature_columns", [])
    all_models = bundle.get("all_models", {})
    if not all_models:
        all_models = {"CatBoost (Tuned)": bundle.get("spend_regressor")}

    X_feat = df[feature_cols].astype(float)
    base = _baseline(X_feat)
    
    comparison_df = df.copy()
    comparison_df["Baseline (180d÷2)"] = base.round(2)

    for m_name, model in all_models.items():
        preds_raw = model.predict(X_feat)
        if bundle.get("residual_target", False):
            val = np.maximum(0.0, np.expm1(np.log1p(base) + preds_raw))
        else:
            val = np.maximum(0.0, np.expm1(preds_raw))
        comparison_df[f"{m_name} (£)"] = val.round(2)

    return comparison_df


# ──────────────────────────────────────────────────────────────────────────────
# Smart Data Inspector & Auto-Repair Engine
# ──────────────────────────────────────────────────────────────────────────────
def smart_data_inspector_and_repair(raw_df: pd.DataFrame, bundle: dict) -> dict:
    """
    Intelligently audits any uploaded dataset:
    - Normalizes column names (handles casing, spaces, underscores, and synonyms)
    - Detects whether dataset is raw transactions or pre-aggregated customer features
    - Detects and cleans text-formatted numbers (e.g. '$1,200', ' 45 ', commas)
    - Detects missing/null values and applies median imputation
    - Flags and clips negative values (e.g. negative recency or spend)
    - Synthesizes any missing model features so predictions NEVER fail
    - Returns audit report, health score, and cleaned ready-to-score dataframe
    """
    repairs_log = []
    warnings_log = []
    audit_rows = []
    
    clean_df = raw_df.copy()
    
    # 1. Column synonym normalization dictionary
    synonyms = {
        "Customer ID": ["customer id", "customer_id", "customerid", "customer", "cust_id", "client_id", "user_id", "id", "custid", "account_id"],
        "Recency": ["recency", "recency_days", "days_since_last", "last_purchase_days", "inactivity_days", "days_last_order", "last_order_days"],
        "Frequency": ["frequency", "freq", "invoices", "orders", "purchases", "num_orders", "transactions", "order_count", "order_frequency"],
        "Monetary": ["monetary", "money", "spend", "total_spend", "total_amount", "amount", "sales", "revenue", "value", "total_monetary", "gross_sales"],
        "Quantity": ["quantity", "qty", "units", "volume", "items_bought"],
        "Price": ["price", "unit_price", "unitprice", "cost", "rate", "price_per_unit"],
        "InvoiceDate": ["invoicedate", "invoice_date", "date", "timestamp", "order_date", "purchase_date", "trans_date"],
        "Invoice": ["invoice", "invoice_no", "invoiceno", "order_id", "order_number", "bill_no", "receipt_id"]
    }
    
    col_map = {}
    for col in clean_df.columns:
        norm_name = re.sub(r'[\s_\-]+', '', str(col).lower())
        matched = False
        for standard_name, syn_list in synonyms.items():
            for s in syn_list:
                norm_syn = re.sub(r'[\s_\-]+', '', s.lower())
                if norm_name == norm_syn:
                    col_map[col] = standard_name
                    matched = True
                    break
            if matched:
                break
    
    if col_map:
        for old_col, new_col in col_map.items():
            if old_col != new_col:
                repairs_log.append(f"Mapped synonymous column `{old_col}` ➔ `{new_col}`")
        clean_df.rename(columns=col_map, inplace=True)

    # 2. Check if this is a Raw Transaction log (has Quantity, Price, InvoiceDate)
    is_transaction_log = all(c in clean_df.columns for c in ["Quantity", "Price"]) and ("Recency" not in clean_df.columns)
    if is_transaction_log:
        repairs_log.append("Detected raw transaction log format. Automatically aggregating into customer RFM & rolling spend features...")
        cid_col = "Customer ID" if "Customer ID" in clean_df.columns else clean_df.columns[0]
        
        # Clean numeric quantity and price
        clean_df["Quantity"] = pd.to_numeric(clean_df["Quantity"].astype(str).str.replace(r'[^\d\.\-]', '', regex=True), errors='coerce').fillna(1)
        clean_df["Price"] = pd.to_numeric(clean_df["Price"].astype(str).str.replace(r'[^\d\.\-]', '', regex=True), errors='coerce').fillna(0)
        # Exclude returns
        clean_df = clean_df[(clean_df["Quantity"] > 0) & (clean_df["Price"] > 0)]
        clean_df["LineTotal"] = clean_df["Quantity"] * clean_df["Price"]
        
        if "InvoiceDate" in clean_df.columns:
            clean_df["InvoiceDate"] = pd.to_datetime(clean_df["InvoiceDate"], errors='coerce')
            max_dt = clean_df["InvoiceDate"].max()
            rec = clean_df.groupby(cid_col)["InvoiceDate"].max().apply(lambda x: (max_dt - x).days if pd.notnull(x) else 30)
            first_dt = clean_df.groupby(cid_col)["InvoiceDate"].min()
            tenure = (max_dt - first_dt).dt.days
        else:
            rec = 30
            tenure = 90
            
        freq = clean_df.groupby(cid_col)["Invoice"].nunique() if "Invoice" in clean_df.columns else clean_df.groupby(cid_col).size()
        mon = clean_df.groupby(cid_col)["LineTotal"].sum()
        
        clean_df = pd.DataFrame({
            "Customer ID": mon.index,
            "Recency": rec,
            "Frequency": freq,
            "Monetary": mon,
            "TenureDays": tenure
        }).reset_index(drop=True)
        repairs_log.append(f"Aggregated {len(raw_df):,} transaction rows into {len(clean_df):,} customer profiles!")

    # 3. Numeric string cleaning, type casting, missing value imputation
    numeric_targets = ["Recency", "Frequency", "Monetary", "UniqueProducts", "AvgBasket",
                       "TotalQuantity", "TenureDays", "R30_Spend", "R60_Spend", "R90_Spend",
                       "R180_Spend", "AvgOrderValue", "SpendPerDay"]
                       
    for col in clean_df.columns:
        orig_dtype = str(raw_df[col].dtype) if col in raw_df.columns else "derived"
        missing_cnt = int(clean_df[col].isna().sum())
        action_note = "Clean"
        
        # Check if column should be numeric
        if col in numeric_targets or np.issubdtype(clean_df[col].dtype, np.number):
            if clean_df[col].dtype == object:
                cleaned_series = clean_df[col].astype(str).str.replace(r'[\$,£,€,\s,"]', '', regex=True)
                cleaned_series = pd.to_numeric(cleaned_series, errors='coerce')
                clean_df[col] = cleaned_series
                action_note = "Cleaned currency/text symbols & converted to numeric"
                repairs_log.append(f"Column `{col}` was stored as text ({orig_dtype}). Converted currency/text to clean numeric float.")
            
            # Missing value imputation
            if clean_df[col].isna().any():
                med_val = float(clean_df[col].median()) if not pd.isna(clean_df[col].median()) else 0.0
                clean_df[col] = clean_df[col].fillna(med_val)
                action_note += f" | Imputed {missing_cnt} missing values with median ({med_val:.1f})"
                repairs_log.append(f"Imputed {missing_cnt} missing values in `{col}` with median ({med_val:.1f}).")
                
            # Negative values check
            if (clean_df[col] < 0).any():
                neg_count = int((clean_df[col] < 0).sum())
                clean_df[col] = clean_df[col].clip(lower=0.0)
                warnings_log.append(f"Column `{col}` contained {neg_count} negative values — clipped to 0.0.")
                action_note += f" | Clipped {neg_count} negative values to 0"
        
        audit_rows.append({
            "Column Name": col,
            "Original Dtype": orig_dtype,
            "Cleaned Dtype": str(clean_df[col].dtype),
            "Missing Count": missing_cnt,
            "Auto-Repair Action": action_note,
            "Sample Value": str(clean_df[col].iloc[0]) if len(clean_df) > 0 else "N/A"
        })

    # 4. Feature Synthesis: Ensure all 13 model features exist
    required_features = bundle.get("feature_columns", [
        'Recency', 'Frequency', 'Monetary', 'UniqueProducts', 'AvgBasket',
        'TotalQuantity', 'TenureDays', 'R30_Spend', 'R60_Spend', 'R90_Spend',
        'R180_Spend', 'AvgOrderValue', 'SpendPerDay'
    ])
    
    missing_req = [f for f in required_features if f not in clean_df.columns]
    if missing_req:
        mon  = clean_df.get("Monetary", pd.Series(100.0, index=clean_df.index)).astype(float)
        freq = clean_df.get("Frequency", pd.Series(2.0, index=clean_df.index)).clip(lower=1).astype(float)
        rec  = clean_df.get("Recency", pd.Series(30.0, index=clean_df.index)).clip(lower=0).astype(float)
        
        if "UniqueProducts" not in clean_df.columns:
            clean_df["UniqueProducts"] = (freq * 2).astype(int)
        if "AvgBasket" not in clean_df.columns:
            clean_df["AvgBasket"] = (mon / freq).round(2)
        if "TotalQuantity" not in clean_df.columns:
            clean_df["TotalQuantity"] = (clean_df["UniqueProducts"] * 3).astype(int)
        if "TenureDays" not in clean_df.columns:
            clean_df["TenureDays"] = (rec + 60).astype(int)
        if "R30_Spend" not in clean_df.columns:
            clean_df["R30_Spend"] = np.where(rec <= 30, (mon * 0.35).round(2), 0.0)
        if "R60_Spend" not in clean_df.columns:
            clean_df["R60_Spend"] = np.where(rec <= 60, (mon * 0.60).round(2), 0.0)
        if "R90_Spend" not in clean_df.columns:
            clean_df["R90_Spend"] = np.where(rec <= 90, (mon * 0.80).round(2), 0.0)
        if "R180_Spend" not in clean_df.columns:
            clean_df["R180_Spend"] = np.where(rec <= 180, mon.round(2), 0.0)
        if "AvgOrderValue" not in clean_df.columns:
            clean_df["AvgOrderValue"] = (mon / freq).round(2)
        if "SpendPerDay" not in clean_df.columns:
            clean_df["SpendPerDay"] = (mon / (clean_df["TenureDays"] + 1)).round(2)
            
        repairs_log.append(f"Auto-synthesized {len(missing_req)} missing model features: {', '.join(missing_req)}")

    # 5. Customer ID presence
    if "Customer ID" not in clean_df.columns:
        clean_df["Customer ID"] = [f"Cust_{i+1:04d}" for i in range(len(clean_df))]
        repairs_log.append("Dataset had no `Customer ID` column — assigned synthetic unique IDs (`Cust_0001`, `Cust_0002`...).")

    # Compute Health Score (0 - 100)
    health_score = max(25, 100 - len(repairs_log) * 7 - len(warnings_log) * 4)
    if not repairs_log and not warnings_log:
        health_score = 100
        
    return {
        "clean_df": clean_df,
        "repairs": repairs_log,
        "warnings": warnings_log,
        "audit_table": pd.DataFrame(audit_rows),
        "health_score": health_score
    }


# ──────────────────────────────────────────────────────────────────────────────
# Cached Model Bundle Loader
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading all ML models & bundle...")
def get_bundle() -> dict:
    return joblib.load(BUNDLE_PATH)


@st.cache_data(show_spinner="Scoring historical customer profiles...")
def get_scored_profiles(_ver: str, model_name: str = "CatBoost (Tuned)") -> pd.DataFrame:
    return score_customers(pd.read_csv(PROFILE_PATH), get_bundle(), model_name=model_name)


# ──────────────────────────────────────────────────────────────────────────────
# Utilities & Styling
# ──────────────────────────────────────────────────────────────────────────────
def fmt(v, prefix="£"): return f"{prefix}{float(v):,.2f}"
def fmt_int(v): return f"{int(v):,}"

def _m(d, *keys, default=0.0):
    for k in keys:
        for kk in (k, k.lower(), k.upper()):
            if kk in d: return float(d[kk])
    return float(default)

def seg_color(s): return SEG_COLORS.get(s, "#1a56db")

def chart_layout(fig, *, legend=False):
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=legend,
        margin=dict(t=40, b=10, l=10, r=10),
        font=dict(family="Inter, sans-serif", size=13),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0", zeroline=False)
    return fig


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1.2rem; max-width: 1400px; }
    .seg-badge {
        display:inline-block; padding:0.25rem 0.85rem;
        border-radius:9999px; font-size:0.84rem; font-weight:600; color:white;
    }
    .kpi-card {
        background:#ffffff; border:1px solid #e5e7eb; border-radius:14px;
        padding:1.1rem 1.4rem; text-align:center; margin-bottom:0.5rem;
        box-shadow:0 1px 3px rgba(0,0,0,0.06);
    }
    .action-box {
        background:#eff6ff; border-left:4px solid #1a56db;
        border-radius:6px; padding:0.85rem 1.1rem; margin:0.7rem 0; font-size:0.94rem;
    }
    div[data-testid="metric-container"] {
        background:#ffffff; border:1px solid #e5e7eb;
        border-radius:12px; padding:0.9rem 1.1rem;
        box-shadow:0 1px 3px rgba(0,0,0,0.05);
    }
    th { background-color:#f8fafc !important; font-weight:600; }
    </style>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# ▌ Page 1 · Overview
# ──────────────────────────────────────────────────────────────────────────────
def page_overview(scored: pd.DataFrame, bundle: dict) -> None:
    st.title("🛍️ Customer LTV & Segmentation Engine")
    st.caption("RFM-based customer segmentation + 90-day spend forecasting · UCI Online Retail II (2009–2011)")

    # ── KPIs ─────────────────────────────────────────────────────────────────
    total   = len(scored)
    n_seg   = scored["Segment_ID"].nunique()
    avg_sp  = scored["Predicted_Next90_Spend"].mean()
    tot_sp  = scored["Predicted_Next90_Spend"].sum()
    n_champ = (scored["Segment"] == "Champions").sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("👥 Total Customers",    fmt_int(total))
    c2.metric("🗂️ Segments",          str(n_seg))
    c3.metric("💷 Avg 90d Forecast",  fmt(avg_sp))
    c4.metric("💰 Total Rev. at Stake", fmt(tot_sp))
    c5.metric("🏆 Champions",         fmt_int(n_champ))

    st.divider()

    # ── Interactive filters ───────────────────────────────────────────────────
    with st.expander("🔧 Filter dashboard data", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        seg_filter = fc1.multiselect(
            "Segments", options=sorted(scored["Segment"].unique()),
            default=list(scored["Segment"].unique()),
        )
        rec_range = fc2.slider("Recency range (days)", 0, int(scored["Recency"].max()),
                               (0, int(scored["Recency"].max())))
        spend_range = fc3.slider("Predicted spend range (£)",
                                 0.0, float(scored["Predicted_Next90_Spend"].quantile(0.99)),
                                 (0.0, float(scored["Predicted_Next90_Spend"].quantile(0.99))))

    filtered = scored[
        scored["Segment"].isin(seg_filter) &
        scored["Recency"].between(*rec_range) &
        scored["Predicted_Next90_Spend"].between(*spend_range)
    ]
    if filtered.empty:
        st.warning("No customers match the current filter selection.")
        return

    # ── Row 1: Segment bar + Spend bar ───────────────────────────────────────
    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        tab_count, tab_pct = st.tabs(["Customer Count", "Market Share (%)"])
        counts = (filtered.groupby("Segment", as_index=False)
                  .agg(Customers=("Customer ID","count"))
                  .sort_values("Customers", ascending=False))
        counts["Percentage"] = (counts["Customers"] / counts["Customers"].sum() * 100).round(1)
        with tab_count:
            fig = chart_layout(px.bar(counts, x="Segment", y="Customers",
                color="Segment", color_discrete_map=SEG_COLORS,
                text_auto=True, title="Customer count by segment"))
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
        with tab_pct:
            fig = chart_layout(px.pie(counts, names="Segment", values="Percentage",
                color="Segment", color_discrete_map=SEG_COLORS,
                title="Segment distribution (%)", hole=0.4))
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        spend_tab1, spend_tab2 = st.tabs(["Average Predicted Spend", "Total Revenue at Stake"])
        spend = (filtered.groupby("Segment", as_index=False)
                 .agg(AvgSpend=("Predicted_Next90_Spend","mean"),
                      TotalSpend=("Predicted_Next90_Spend","sum"))
                 .sort_values("AvgSpend", ascending=False))
        with spend_tab1:
            fig = chart_layout(px.bar(spend, x="Segment", y="AvgSpend",
                color="Segment", color_discrete_map=SEG_COLORS,
                text_auto=".0f", title="Average predicted next-90-day spend (£)"))
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
        with spend_tab2:
            fig = chart_layout(px.bar(spend, x="Segment", y="TotalSpend",
                color="Segment", color_discrete_map=SEG_COLORS,
                text_auto=".0f", title="Total revenue at stake by segment (£)"))
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Interactive RFM explorer ──────────────────────────────────────
    st.subheader("🔭 Interactive RFM Explorer")
    ec1, ec2, ec3 = st.columns(3)
    x_axis = ec1.selectbox("X axis", ["Recency","Frequency","Monetary","R90_Spend","R180_Spend",
                                       "TenureDays","AvgOrderValue","SpendPerDay"], index=0)
    y_axis = ec2.selectbox("Y axis", ["Monetary","Recency","Frequency","R90_Spend","R180_Spend",
                                       "TenureDays","AvgOrderValue","SpendPerDay"], index=0)
    size_by = ec3.selectbox("Bubble size", ["Frequency","Predicted_Next90_Spend","Monetary",
                                             "TotalQuantity"], index=0)

    sample = filtered.sample(min(2500, len(filtered)), random_state=42)
    fig = px.scatter(
        sample, x=x_axis, y=y_axis, size=size_by, size_max=20, opacity=0.65,
        color="Segment", color_discrete_map=SEG_COLORS,
        hover_data=["Customer ID","Recency","Frequency","Monetary","Predicted_Next90_Spend"],
        title=f"{x_axis} vs {y_axis} (bubble size = {size_by})",
    )
    chart_layout(fig, legend=True)
    st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Heatmap of RFM correlations ───────────────────────────────────
    st.subheader("🌡️ Feature Correlation Heatmap")
    corr_cols = [c for c in ["Recency","Frequency","Monetary","UniqueProducts",
                              "TenureDays","R30_Spend","R90_Spend","R180_Spend",
                              "AvgOrderValue","Predicted_Next90_Spend"]
                 if c in filtered.columns]
    corr = filtered[corr_cols].corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdBu", zmid=0, text=corr.values,
        texttemplate="%{text}", hovertemplate="%{x} × %{y}: %{z}<extra></extra>",
    ))
    fig.update_layout(title="Feature correlation matrix (Pearson r)", height=450,
                      plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(t=40, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

    # ── Row 4: Segment profiles table + box plots ─────────────────────────────
    st.subheader("📋 Segment Profiles & Action Strategy")
    tab_tbl, tab_box = st.tabs(["Summary Table", "Boxplot Distributions"])
    with tab_tbl:
        tbl = (filtered.groupby(["Segment","Recommended_Action"], as_index=False)
               .agg(Customers=("Customer ID","count"),
                    AvgRecency=("Recency","mean"),
                    AvgFrequency=("Frequency","mean"),
                    AvgMonetary=("Monetary","mean"),
                    AvgPredicted=("Predicted_Next90_Spend","mean"),
                    TotalRevenue=("Predicted_Next90_Spend","sum"))
               .sort_values("AvgPredicted", ascending=False))
        for c in ["AvgRecency","AvgFrequency","AvgMonetary","AvgPredicted","TotalRevenue"]:
            tbl[c] = tbl[c].round(1)
        tbl.columns = ["Segment","Recommended Action","Customers",
                        "Avg Recency (d)","Avg Frequency","Avg Spend (£)",
                        "Avg 90d Forecast (£)","Total Rev. (£)"]
        st.dataframe(tbl, hide_index=True, use_container_width=True)
    with tab_box:
        box_metric = st.selectbox("Distribution metric",
            ["Predicted_Next90_Spend","Monetary","Frequency","Recency"], key="box_met")
        fig = px.box(filtered, x="Segment", y=box_metric, color="Segment",
                     color_discrete_map=SEG_COLORS,
                     points="outliers", notched=True,
                     title=f"{box_metric} distribution across segments")
        chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# ▌ Page 2 · Customer Lookup
# ──────────────────────────────────────────────────────────────────────────────
def page_customer_lookup(scored: pd.DataFrame, bundle: dict) -> None:
    st.title("🔍 Customer Lookup")
    st.write("Search any historical customer — view their assigned segment, RFM diagnostics, forecast breakdown, and peer rank.")

    cid_input = st.text_input("Customer ID", placeholder="e.g. 12347, 14646, 17850…")

    if not cid_input.strip():
        st.info("👆 Enter a Customer ID to inspect their profile.")
        sample_ids = scored.sample(min(8, len(scored)), random_state=12)["Customer ID"].tolist()
        st.caption("Try one of these IDs: " + "  ·  ".join(str(x) for x in sample_ids))
        return

    match = scored[scored["Customer ID"].astype(str) == str(cid_input.strip())]
    if match.empty:
        # Try numeric conversion
        try:
            num_id = int(float(cid_input.strip()))
            match = scored[scored["Customer ID"] == num_id]
        except Exception:
            pass

    if match.empty:
        st.warning(f"Customer **{cid_input}** was not found in the historical database.")
        return

    row   = match.iloc[0]
    seg   = str(row["Segment"])
    color = seg_color(seg)

    st.markdown(f"### Customer {row['Customer ID']}")
    st.markdown(f'<span class="seg-badge" style="background:{color}">{seg}</span>',
                unsafe_allow_html=True)
    st.markdown("")

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Predicted 90d Spend",   fmt(row["Predicted_Next90_Spend"]))
    k2.metric("Historical Baseline",   fmt(row.get("Baseline_90d_Spend", 0)))
    k3.metric("Historical Spend",      fmt(row["Monetary"]))
    k4.metric("Frequency",             fmt_int(row["Frequency"]))

    st.markdown(
        f'<div class="action-box">💡 <strong>Recommended Marketing Action:</strong> {row["Recommended_Action"]}</div>',
        unsafe_allow_html=True,
    )

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown("**Customer behavioral profile**")
        profile_rows = [
            ("Recency (days)",       round(float(row["Recency"]), 0)),
            ("Frequency (orders)",   int(row["Frequency"])),
            ("Total spend (£)",      fmt(row["Monetary"])),
            ("Unique products",      int(row.get("UniqueProducts", 0))),
            ("Tenure (days active)", int(row.get("TenureDays", 0))),
            ("Avg order value",      fmt(row.get("AvgOrderValue", 0))),
            ("Last 30d spend",       fmt(row.get("R30_Spend", 0))),
            ("Last 90d spend",       fmt(row.get("R90_Spend", 0))),
            ("Last 180d spend",      fmt(row.get("R180_Spend", 0))),
        ]
        st.dataframe(pd.DataFrame(profile_rows, columns=["Metric","Value"]),
                     hide_index=True, use_container_width=True)

    with col_r:
        st.markdown("**Forecast composition**")
        baseline = float(row.get("Baseline_90d_Spend", 0))
        ml_adj   = float(row.get("ML_Adjusted_90d_Spend", row["Predicted_Next90_Spend"]))
        final    = float(row["Predicted_Next90_Spend"])
        ml_delta = ml_adj - baseline

        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "total"],
            x=["180d Baseline", "ML Delta", "Final Blended Forecast"],
            y=[baseline, ml_delta, final],
            text=[fmt(baseline), fmt(ml_delta, prefix="+" if ml_delta >= 0 else ""), fmt(final)],
            textposition="outside",
            connector={"line": {"color": "#9ca3af"}},
            increasing={"marker": {"color": "#1a56db"}},
            decreasing={"marker": {"color": "#ef4444"}},
            totals={"marker": {"color": "#16a34a"}},
        ))
        fig.update_layout(title="Forecast waterfall breakdown",
                          plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    # Segment percentile
    st.subheader("📍 Segment context")
    peers = scored[scored["Segment"] == seg]
    pct   = (peers["Predicted_Next90_Spend"] < row["Predicted_Next90_Spend"]).mean() * 100

    prog_col, stat_col = st.columns([3, 1])
    prog_col.progress(pct / 100)
    stat_col.markdown(f"**{pct:.0f}th percentile** in {seg}")

    fig = px.histogram(peers, x="Predicted_Next90_Spend", nbins=40,
                       color_discrete_sequence=[color],
                       title=f"Predicted spend distribution — {seg} peers",
                       labels={"Predicted_Next90_Spend": "Predicted 90d Spend (£)"})
    fig.add_vline(x=row["Predicted_Next90_Spend"], line_color="#ef4444",
                  line_width=2, annotation_text=f"Customer {row['Customer ID']}",
                  annotation_position="top right")
    chart_layout(fig)
    st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# ▌ Page 3 · Train Your Model (Live In-App Training)
# ──────────────────────────────────────────────────────────────────────────────
def page_train_model() -> None:
    st.title("🔬 Train Your Model")
    st.write("Tune K-Means segmentation and CatBoost regression parameters, train live, and view instant performance curves.")

    if not PROFILE_PATH.exists():
        st.error("Customer profiles dataset not found.")
        return
    profiles = pd.read_csv(PROFILE_PATH)

    st.subheader("1️⃣ Data Source")
    st.info(f"Using built-in dataset: **{len(profiles):,} historical customer profiles**")

    st.divider()
    st.subheader("2️⃣ K-Means Clustering Parameters")
    sc1, sc2, sc3 = st.columns(3)
    k = sc1.slider("Clusters (k)", 2, 8, 4)
    km_init = sc2.selectbox("Initialization", ["k-means++", "random"])
    km_maxiter = sc3.slider("Max iterations", 100, 500, 300, step=50)

    st.divider()
    st.subheader("3️⃣ CatBoost Spend Regressor Parameters")
    rc1, rc2, rc3 = st.columns(3)
    rc4, rc5 = st.columns(2)
    cb_iters = rc1.slider("Iterations", 50, 400, 150, step=25)
    cb_depth = rc2.slider("Tree depth", 2, 10, 5)
    cb_lr    = rc3.slider("Learning rate", 0.01, 0.25, 0.05, step=0.01)
    cb_l2    = rc4.slider("L2 regularization", 1.0, 15.0, 5.0, step=0.5)
    test_sz  = rc5.slider("Test set size", 0.1, 0.4, 0.2, step=0.05)

    feat_options = ['Recency', 'Frequency', 'Monetary', 'UniqueProducts', 'AvgBasket',
                    'TotalQuantity', 'TenureDays', 'R30_Spend', 'R60_Spend', 'R90_Spend',
                    'R180_Spend', 'AvgOrderValue', 'SpendPerDay']

    if st.button("🚀 Train Model Live", type="primary", use_container_width=True):
        progress = st.progress(0, text="Preprocessing data…")
        
        # 1. Clustering
        X_rfm = np.log1p(profiles[['Recency', 'Frequency', 'Monetary']].clip(lower=0).astype(float))
        scaler = StandardScaler()
        Z = scaler.fit_transform(X_rfm)
        progress.progress(25, text="Fitting K-Means…")

        km = KMeans(n_clusters=k, init=km_init, max_iter=km_maxiter, random_state=42, n_init=10)
        labels = km.fit_predict(Z)
        sil = silhouette_score(Z, labels, sample_size=3000, random_state=42)
        progress.progress(50, text="Fitting CatBoost Regressor…")

        # 2. Regression
        X = profiles[feat_options].astype(float).values
        y = profiles["Predicted_Next90_Spend"].astype(float).values
        y_log = np.log1p(y)

        X_tr, X_te, y_tr, y_te = train_test_split(X, y_log, test_size=test_sz, random_state=42)
        
        cb = CatBoostRegressor(
            iterations=cb_iters, depth=cb_depth, learning_rate=cb_lr,
            l2_leaf_reg=cb_l2, loss_function="RMSE", verbose=False,
            random_seed=42, thread_count=1
        )
        cb.fit(X_tr, y_tr)
        progress.progress(85, text="Evaluating metrics…")

        y_pred_te = np.maximum(0.0, np.expm1(cb.predict(X_te)))
        actuals_te = np.expm1(y_te)

        mae_te = mean_absolute_error(actuals_te, y_pred_te)
        rmse_te = float(np.sqrt(mean_squared_error(actuals_te, y_pred_te)))
        r2_te = r2_score(actuals_te, y_pred_te)
        progress.progress(100, text="Done!")

        st.success("✅ Training completed successfully!")
        
        # Metrics display
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Silhouette Score (k)", f"{sil:.4f}")
        m2.metric("Test MAE", fmt(mae_te))
        m3.metric("Test RMSE", fmt(rmse_te))
        m4.metric("Test R²", f"{r2_te:.4f}")

        # Feature importance chart
        fi = pd.DataFrame({
            "Feature": feat_options,
            "Importance": cb.get_feature_importance()
        }).sort_values("Importance", ascending=True)
        fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                     title="Feature importance (CatBoost)", color="Importance",
                     color_continuous_scale="Blues")
        chart_layout(fig)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# ▌ Page 4 · What-If Forecast Simulator
# ──────────────────────────────────────────────────────────────────────────────
def page_whatif(scored: pd.DataFrame, bundle: dict) -> None:
    st.title("🎯 What-If Forecast Simulator")
    st.write("Simulate any customer profile with interactive sliders — see their segment, action, and spend forecast update in real-time.")

    profiles = pd.read_csv(PROFILE_PATH)
    def q(col, val): return float(profiles[col].quantile(val)) if col in profiles.columns else 0.0

    st.markdown("### 🎛️ Customer Profile Sliders")
    c1, c2, c3 = st.columns(3)
    recency   = c1.slider("Recency (days)", 0, int(q("Recency", 0.99)), int(q("Recency", 0.5)))
    frequency = c2.slider("Frequency (orders)", 1, int(max(q("Frequency", 0.99), 2)), int(q("Frequency", 0.5)))
    monetary  = c3.slider("Monetary spend (£)", 0.0, float(q("Monetary", 0.99)), float(q("Monetary", 0.5)), step=10.0)

    c4, c5, c6 = st.columns(3)
    uniq_prod = c4.slider("Unique products", 1, int(max(q("UniqueProducts", 0.99), 2)), int(q("UniqueProducts", 0.5)))
    tenure    = c5.slider("Tenure (days)", 0, int(q("TenureDays", 0.99)), int(q("TenureDays", 0.5)))
    avg_bsk   = c6.slider("Avg basket size (£)", 0.0, float(q("AvgBasket", 0.99)), float(q("AvgBasket", 0.5)), step=5.0)

    # Derived
    sim_df = pd.DataFrame([{
        "Customer ID": "Simulated_Cust",
        "Recency": recency, "Frequency": frequency, "Monetary": monetary,
        "UniqueProducts": uniq_prod, "AvgBasket": avg_bsk,
        "TotalQuantity": uniq_prod * 3, "TenureDays": tenure,
        "R30_Spend": monetary * 0.3 if recency <= 30 else 0.0,
        "R60_Spend": monetary * 0.5 if recency <= 60 else 0.0,
        "R90_Spend": monetary * 0.8 if recency <= 90 else 0.0,
        "R180_Spend": monetary if recency <= 180 else 0.0,
        "AvgOrderValue": monetary / max(frequency, 1),
        "SpendPerDay": monetary / max(tenure + 1, 1),
    }])

    res = score_customers(sim_df, bundle)
    pred_seg = res.iloc[0]["Segment"]
    pred_sp  = float(res.iloc[0]["Predicted_Next90_Spend"])
    baseline = float(res.iloc[0]["Baseline_90d_Spend"])
    action   = res.iloc[0]["Recommended_Action"]
    color    = seg_color(pred_seg)

    st.divider()
    res1, res2, res3 = st.columns(3)
    res1.markdown(
        f'<div class="kpi-card"><div style="font-size:0.8rem;color:#6b7280;margin-bottom:0.2rem">Predicted Segment</div>'
        f'<span class="seg-badge" style="background:{color}">{pred_seg}</span></div>',
        unsafe_allow_html=True
    )
    res2.metric("Predicted 90d Spend", fmt(pred_sp))
    res3.metric("Historical Baseline", fmt(baseline))

    st.markdown(f'<div class="action-box">💡 <strong>Action:</strong> {action}</div>', unsafe_allow_html=True)

    # Gauge Chart
    max_gauge = max(float(scored["Predicted_Next90_Spend"].quantile(0.99)), pred_sp * 1.2, 500.0)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pred_sp,
        delta={"reference": baseline, "valueformat": ".0f",
               "increasing": {"color": "#16a34a"}, "decreasing": {"color": "#ef4444"}},
        number={"prefix": "£", "valueformat": ".0f"},
        gauge={
            "axis": {"range": [0, max_gauge], "tickprefix": "£"},
            "bar":  {"color": color},
            "steps": [
                {"range": [0, max_gauge * 0.33], "color": "#f3f4f6"},
                {"range": [max_gauge * 0.33, max_gauge * 0.66], "color": "#e5e7eb"},
                {"range": [max_gauge * 0.66, max_gauge], "color": "#d1d5db"},
            ],
            "threshold": {"line": {"color": "#ef4444", "width": 3}, "thickness": 0.75, "value": baseline},
        },
        title={"text": "Predicted 90-Day Spend vs Baseline"},
    ))
    fig.update_layout(height=300, margin=dict(t=40, b=10, l=30, r=30), paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# ▌ Page 5 · 🛡️ Smart Data Inspector & Auto-Predictor (Handles ANY dataset)
# ──────────────────────────────────────────────────────────────────────────────
def page_smart_inspector(bundle: dict) -> None:
    st.title("🛡️ Smart Data Inspector & Auto-Predictor")
    st.write(
        "Upload **ANY** customer or transaction dataset (clean or dirty, raw transactions or RFM). "
        "The engine automatically inspects dtypes, syntax, missing values, cleans dirty formatting, "
        "synthesizes missing features, and predicts 90-day spend using any of the **7 trained models**."
    )

    # 1. Model Selector
    all_models = bundle.get("all_models", {})
    model_options = list(all_models.keys()) if all_models else ["CatBoost (Tuned)"]
    if "Ensemble Blend" not in model_options:
        model_options.append("Ensemble Blend")

    col_m1, col_m2 = st.columns([2, 1])
    selected_model = col_m1.selectbox("Select ML Model for Prediction:", model_options, index=0)
    compare_all = col_m2.checkbox("⚡ Compare all 5 models side-by-side", value=False)

    st.divider()

    # 2. File Uploader or Demo Loader
    col_u1, col_u2 = st.columns([3, 1])
    uploaded_file = col_u1.file_uploader(
        "Upload CSV, Excel (.xlsx, .xls), or TSV file",
        type=["csv", "xlsx", "xls", "tsv", "txt"]
    )
    
    with col_u2:
        st.markdown("**Quick demo datasets:**")
        load_dirty = st.button("🧪 Load Dirty / Messy CSV")
        load_clean = st.button("📋 Load Clean Sample CSV")

    raw_df = None
    if uploaded_file is not None:
        try:
            fname = uploaded_file.name.lower()
            if fname.endswith(".xlsx") or fname.endswith(".xls"):
                raw_df = pd.read_excel(uploaded_file)
            elif fname.endswith(".tsv") or fname.endswith(".tab"):
                raw_df = pd.read_csv(uploaded_file, sep="\t")
            else:
                raw_df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded `{uploaded_file.name}`: **{len(raw_df):,} rows, {len(raw_df.columns)} columns**.")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return
    elif load_dirty:
        raw_df = pd.DataFrame({
            "Cust_ID": [101, 102, 103, 104, 105, 106],
            "Days_Since_Last": [12, -4, 180, 45, np.nan, 300],
            "num_orders": [" 3 ", "12", "1", "$4", "6", " 2 "],
            "Total_Spend": ["$1,250.00", "£4,500.50", " 150 ", "-50", "$2,100", "850.00"],
            "Category_Notes": ["VIP", "VIP", "Regular", "Active", "At-Risk", "Hibernating"]
        })
        st.info("Loaded demo **Dirty / Messy Dataset** (has '$' currency symbols, strings in numeric columns, negative days, missing values, and renamed columns).")
    elif load_clean:
        sample_path = ROOT / "data" / "sample_customer_features.csv"
        if sample_path.exists():
            raw_df = pd.read_csv(sample_path)
            st.info("Loaded demo **Clean Customer Features Sample**.")

    if raw_df is None:
        st.info("👆 Upload your dataset above or click one of the demo buttons to see the diagnostic & prediction engine in action.")
        return

    # 3. Run Smart Data Inspector & Auto-Repair Engine
    with st.spinner("Analyzing schema, dtypes, and repairing anomalies..."):
        diag = smart_data_inspector_and_repair(raw_df, bundle)

    clean_df     = diag["clean_df"]
    repairs_log  = diag["repairs"]
    warnings_log = diag["warnings"]
    audit_table  = diag["audit_table"]
    health_score = diag["health_score"]

    # 4. Display Health Score & Diagnostic Cards
    st.subheader("🩺 Data Quality Health Check")
    h1, h2, h3 = st.columns([1, 2, 2])
    
    score_color = "#16a34a" if health_score >= 80 else ("#d97706" if health_score >= 50 else "#ef4444")
    h1.markdown(
        f'<div class="kpi-card"><div style="font-size:0.8rem;color:#6b7280">Health Score</div>'
        f'<div style="font-size:2rem;font-weight:700;color:{score_color}">{health_score}/100</div>'
        f'<div style="font-size:0.75rem;color:#6b7280">Auto-Repaired & Ready</div></div>',
        unsafe_allow_html=True
    )
    
    with h2:
        st.markdown(f"**🛠️ Auto-Repairs Applied ({len(repairs_log)}):**")
        if repairs_log:
            for r in repairs_log:
                st.markdown(f"✓ *{r}*")
        else:
            st.success("Dataset schema is completely clean — no repairs needed.")

    with h3:
        st.markdown(f"**⚠️ Data Quality Warnings ({len(warnings_log)}):**")
        if warnings_log:
            for w in warnings_log:
                st.warning(w)
        else:
            st.info("No critical anomalies found.")

    with st.expander("🔍 View Complete Dtype & Schema Diagnostic Audit", expanded=False):
        st.dataframe(audit_table, hide_index=True, use_container_width=True)

    # 5. Score Dataset using Selected Model
    st.divider()
    st.subheader(f"📦 Scoring Results — {selected_model}")
    
    with st.spinner(f"Scoring {len(clean_df):,} customers using {selected_model}…"):
        scored_data = score_customers(clean_df, bundle, model_name=selected_model)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers Scored", fmt_int(len(scored_data)))
    k2.metric("Total Projected 90d Spend", fmt(scored_data["Predicted_Next90_Spend"].sum()))
    k3.metric("Avg Predicted Spend", fmt(scored_data["Predicted_Next90_Spend"].mean()))
    k4.metric("Champions Identified", fmt_int((scored_data["Segment"] == "Champions").sum()))

    # Visualizations
    v1, v2 = st.columns(2)
    with v1:
        seg_dist = scored_data.groupby("Segment").size().reset_index(name="Count")
        fig = px.pie(seg_dist, names="Segment", values="Count", color="Segment",
                     color_discrete_map=SEG_COLORS, title="Predicted segment distribution", hole=0.35)
        st.plotly_chart(fig, use_container_width=True)
    with v2:
        fig = px.histogram(scored_data, x="Predicted_Next90_Spend", color="Segment",
                           color_discrete_map=SEG_COLORS, nbins=30,
                           title="Predicted next-90-day spend distribution (£)")
        chart_layout(fig, legend=True)
        st.plotly_chart(fig, use_container_width=True)

    # Display Scored Table
    preview_cols = [c for c in [
        "Customer ID", "Segment", "Predicted_Next90_Spend",
        "Baseline_90d_Spend", "ML_Adjusted_90d_Spend", "Recommended_Action"
    ] if c in scored_data.columns]
    
    st.markdown("**Scored customer table preview:**")
    st.dataframe(scored_data[preview_cols].head(100), hide_index=True, use_container_width=True)

    # Multi-model side-by-side comparison if toggled
    if compare_all:
        st.subheader("⚡ Multi-Model Side-by-Side Comparison")
        st.write("Compare predictions across all 5 candidate models simultaneously for each customer:")
        multi_scored = score_all_candidate_models(clean_df, bundle)
        model_cols = [c for c in multi_scored.columns if "£" in c or c in ["Customer ID", "Baseline (180d÷2)"]]
        st.dataframe(multi_scored[model_cols].head(100), hide_index=True, use_container_width=True)
        
        # Download multi-model comparison
        st.download_button(
            "⬇️ Download Multi-Model Comparison CSV",
            data=multi_scored.to_csv(index=False).encode("utf-8"),
            file_name="multi_model_prediction_comparison.csv",
            mime="text/csv"
        )

    # Downloads
    d1, d2 = st.columns(2)
    d1.download_button(
        f"⬇️ Download Scored Dataset ({selected_model})",
        data=scored_data.to_csv(index=False).encode("utf-8"),
        file_name=f"scored_customers_{selected_model.replace(' ', '_')}.csv",
        mime="text/csv"
    )
    d2.download_button(
        "⬇️ Download Repaired Clean Data",
        data=clean_df.to_csv(index=False).encode("utf-8"),
        file_name="clean_repaired_dataset.csv",
        mime="text/csv"
    )


# ──────────────────────────────────────────────────────────────────────────────
# ▌ Page 6 · Methodology & Evaluation
# ──────────────────────────────────────────────────────────────────────────────
def page_methodology(bundle: dict) -> None:
    st.title("📋 Methodology & Evaluation")

    tab_ov, tab_mod, tab_metrics, tab_lim = st.tabs([
        "Project Architecture", "All 7 Trained Models", "Evaluation Scores", "Limitations"
    ])

    with tab_ov:
        st.markdown("""
        ## Customer Lifetime Value & Segmentation Engine

        **Problem Framing:** Group customers by historical purchase behavior and predict how much each one will spend in the next 90 days.

        | Step | Technical Implementation |
        |---|---|
        | 1 · Data Understanding & Cleaning | Removed cancellations ('C' invoices), missing Customer IDs, and negative quantities |
        | 2 · Feature Engineering | Built RFM, rolling spend windows (30/60/90/180 days), basket sizes, order values, and tenure |
        | 3 · Leakage-Free Target | Cutoff date 90 days before max date. Features use strictly historical data; target is future spend |
        | 4 · Unsupervised Segmentation | StandardScaler on log-transformed RFM, followed by K-Means (k=4) and Hierarchical Clustering |
        | 5 · Supervised Spend Models | Trained and compared Baseline (Mean), Linear Regression, Random Forest, Gradient Boosting, CatBoost |
        | 6 · Hyperparameter Tuning | GridSearchCV 3-fold cross-validation on CatBoost |
        """)

    with tab_mod:
        st.markdown("""
        ## All 7 Trained Models in Deployment
        The project trains and preserves **all 7 models** across unsupervised and supervised categories:

        ### 🧠 Unsupervised Segmentation Models
        1. **K-Means Clustering ($k=4$):** Primary segmentation model (`kmeans_model.joblib`), silhouette score: **0.378**.
        2. **Hierarchical (Agglomerative) Clustering ($k=4$):** Benchmark hierarchical clusterer (`hierarchical_model.joblib`).

        ### 🎯 Supervised Regression Models
        3. **CatBoost Regressor (Tuned):** Primary spend forecasting engine (`model_catboost_tuned.joblib`).
        4. **Random Forest Regressor:** Non-linear ensemble model (`model_random_forest.joblib`).
        5. **Gradient Boosting Regressor:** Sequential boosting tree regressor (`model_gradient_boosting.joblib`).
        6. **Linear Regression:** Parametric statistical benchmark (`model_linear_regression.joblib`).
        7. **Baseline (Mean DummyRegressor):** Null benchmark predicting historical average (`model_baseline_mean.joblib`).
        """)

    with tab_metrics:
        st.markdown("## Test Set Model Comparison")
        comp_path = ROOT / "reports" / "test_model_comparison.csv"
        if comp_path.exists():
            comp_df = pd.read_csv(comp_path)
            st.dataframe(comp_df, hide_index=True, use_container_width=True)

        fi_path = ROOT / "reports" / "feature_importance.csv"
        if fi_path.exists():
            st.subheader("Feature Importance (CatBoost)")
            fi = pd.read_csv(fi_path).sort_values("Importance", ascending=True)
            fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="Blues")
            chart_layout(fig)
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab_lim:
        st.warning("""
        **Model Limitations & Considerations:**
        - **Historical Context:** Trained on 2009–2011 UK retail data. Not directly suitable for live retail forecasting without calibration on modern data.
        - **Gross Spend Metric:** Predicts gross sales value; returns/cancellations are excluded during data preparation.
        - **Cold Start:** Customers with zero prior transaction history cannot be scored until their first order.
        """)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ──────────────────────────────────────────────────────────────────────────────
def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## 🛍️ Customer LTV Engine")
        st.caption("NextGen Institute · ML Final Project")
        st.divider()

        page = st.radio(
            "Navigate",
            ["📊 Overview",
             "🔍 Customer Lookup",
             "🔬 Train Your Model",
             "🎯 What-If Forecast",
             "🛡️ Smart Data Inspector & Predictor",
             "📋 Methodology"],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown(
            "**Dataset:** UCI Online Retail II  \n"
            "**~1M transactions** (2009–2011)  \n"
            "**Models Deployed:** 7 models saved  \n"
            "**Horizon:** 90 days"
        )
    return page


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap Execution
# ──────────────────────────────────────────────────────────────────────────────
inject_css()

bundle          = None
scored_profiles = None

if BUNDLE_PATH.exists() and PROFILE_PATH.exists():
    try:
        bundle          = get_bundle()
        scored_profiles = get_scored_profiles(bundle.get("bundle_version", "1.1.0"))
    except Exception as exc:
        st.sidebar.warning(f"Bundle load notice: {exc}")

page = render_sidebar()

if page == "📊 Overview":
    if scored_profiles is None:
        st.error("Model assets not found.")
    else:
        page_overview(scored_profiles, bundle)

elif page == "🔍 Customer Lookup":
    if scored_profiles is None:
        st.error("Model assets not found.")
    else:
        page_customer_lookup(scored_profiles, bundle)

elif page == "🔬 Train Your Model":
    page_train_model()

elif page == "🎯 What-If Forecast":
    if bundle is None:
        st.error("Pre-trained bundle required for the simulator.")
    else:
        page_whatif(scored_profiles, bundle)

elif page == "🛡️ Smart Data Inspector & Predictor":
    if bundle is None:
        st.error("Model bundle required for scoring.")
    else:
        page_smart_inspector(bundle)

else:  # Methodology
    if bundle is None:
        bundle = {}
    page_methodology(bundle)