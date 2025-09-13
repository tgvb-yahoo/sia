
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="SIA App — v21 (Default improvement package + HI fix)", layout="wide")
st.title("Social Impact Assessment — v21")
st.info("Upload your data file to calculate different measures of Social Impact Assessment")

# ============ Helpers ============
def format_label(x): 
    return str(x).replace("_", " ")

def normalize_01(x, lo, hi):
    return np.clip((x - lo) / (hi - lo) if (hi is not None and lo is not None and hi != lo) else 0.0, 0, 1)

def _shift_nonnegative(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return arr
    mn = np.nanmin(arr)
    return arr - mn if mn < 0 else arr

def gini_coefficient(x):
    arr = np.asarray(x, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan
    arr = _shift_nonnegative(arr)
    if np.all(arr == 0):
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    cum = np.cumsum(arr)
    g = (n + 1 - 2 * np.sum(cum) / cum[-1]) / n
    return float(g)

def lorenz_points(values):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    v = _shift_nonnegative(v)
    v = np.sort(v)
    csum = np.cumsum(v)
    total = csum[-1] if csum.size else 0.0
    if total == 0:
        x = np.linspace(0, 1, len(v) + 1)
        y = np.zeros_like(x)
        return x, y
    x = np.concatenate(([0.0], np.arange(1, len(v)+1)/len(v)))
    y = np.concatenate(([0.0], csum/total))
    return x, y

def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf

def make_fig_download(label, fig, base_name):
    st.download_button(
        label=label,
        data=fig_to_png_bytes(fig),
        file_name=f"{base_name.replace(' ', '-')}.png",
        mime="image/png"
    )

def df_to_png_bytes(df, title=None):
    fig, ax = plt.subplots(figsize=(min(12, 2 + 0.15*len(df.columns)), 0.5 + 0.35*min(25, len(df))))
    ax.axis('off')
    tbl = ax.table(cellText=df.values, colLabels=[format_label(c) for c in df.columns],
                   loc='center', cellLoc='left')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.2)
    if title:
        ax.set_title(format_label(title))
    return fig_to_png_bytes(fig)

def make_df_downloads(df, base_name, add_png=True):
    st.download_button(
        label=f"Download {base_name} (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"{base_name.replace(' ', '-')}.csv",
        mime="text/csv"
    )
    if add_png:
        st.download_button(
            label=f"Download {base_name} (PNG)",
            data=df_to_png_bytes(df, title=base_name),
            file_name=f"{base_name.replace(' ', '-')}.png",
            mime="image/png"
        )

# ---------- Plain-language "copy" helpers ----------
def copy_block(title, text):
    st.markdown(f"**Copy explanation — {title}**")
    st.code(text, language=None)

def formulas_reference():
    with st.expander("Show formulas used in calculations"):
        st.markdown(r"""
**Normalization (0–1):**  
\(
x_i^\*=\frac{x_i - \min(x)}{\max(x)-\min(x)} \quad\) and reverse: \(\quad 1- x_i^\*
\)

**Social Impact Score (SIS):**  
\(
SIS = \sum_{i=1}^k w_i\,x_i \quad \text{with } \sum w_i = 1,\; x_i \in [0,1]
\)

**Health Index (example components):**  
\(
HI = \frac{w_W\,DW + w_T\,TO + w_K\,SK + w_A\,A}{\sum_j w_j}\)
, where access \(A = 1 - \frac{d_h - d_{\min}}{d_{\max} - d_{\min}}.\)

**Asset Index:**  
\(
AI = \frac{w_L \cdot \text{LivestockScore} + w_{LA} \cdot \text{LandScore}}{w_L + w_{LA}}
\)

**Gini (pairwise form):**  
\(
G = \frac{\sum_{i=1}^n\sum_{j=1}^n |y_i - y_j|}{2n^2\bar{y}} \quad \text{and} \quad
G = 1 - 2\int_0^1 L(p)\,dp
\)

**Proportion test (one sided):** Testing \(H_0: p_b \ge p_a\) vs \(H_a: p_b < p_a\)  
\(
z = \frac{p_b - p_a}{\sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_b}+\frac{1}{n_a}\right)}}, \;
\hat{p} = \frac{x_b + x_a}{n_b + n_a}
\)

**After scenario for bottom 40 percent:**  
Let \(m_g = 1+\Delta_g/100\), \(m_{40}=1+\Delta_{40}/100\).  
\(
y_i' = \begin{cases}
y_i \cdot m_g \cdot m_{40}, & i \in \text{bottom }40\% \\
y_i \cdot m_g, & \text{otherwise}
\end{cases}
\)
""")
        copy_block(
            "All formulas (plain language)",
            "Normalization: scale each numeric variable between 0 and 1 using its min and max in the data. "
            "SIS: weighted average of 0–1 indicator scores (weights sum to 1). "
            "Health Index: weighted average of drinking water, toilet, separate kitchen, and access (closer hospital = higher access score). "
            "Asset Index: weighted mix of a livestock score and a land score (each built from normalized components). "
            "Gini: inequality measure from 0 (equal) to 1 (unequal), computed from the Lorenz curve. "
            "Proportion test: pooled one sided z test comparing before vs after proportions. "
            "After scenario (bottom 40 percent): apply an economy wide percentage change to everyone and an extra percentage change to the poorest 40 percent only."
        )

# ============ UI ============
formulas_reference()

mode = st.radio("Data Mode", ["Use scenarios (sliders/toggles)", "Single CSV with After columns", "Two CSVs: Before + After"], index=0, horizontal=True)

# ---- Default improvement package ----
with st.expander("Default improvement package (applied only where After data is missing)"):
    enable_pkg = st.checkbox("Enable default improvements", True)
    col1, col2 = st.columns(2)
    with col1:
        def_inc_global = st.number_input("Income - economy wide change (%)", -100, 500, 5, 1)
        def_inc_b40 = st.number_input("Income - extra for bottom 40% (%)", -100, 500, 15, 1)
        def_sav_global = st.number_input("Savings - economy wide change (%)", -100, 500, 8, 1)
        def_sav_b40 = st.number_input("Savings - extra for bottom 40% (%)", -100, 500, 20, 1)
        def_access_red = st.number_input("Health access - reduce distance by (%)", 0, 100, 20, 1)
    with col2:
        def_livestock_mult = st.number_input("Assets - livestock multiplier (×)", 0.0, 5.0, 1.2, 0.05)
        def_land_mult = st.number_input("Assets - land multiplier (×)", 0.0, 5.0, 1.1, 0.05)
        def_set_water = st.checkbox("Set drinking water to 1 in After", True)
        def_set_toilet = st.checkbox("Set toilet to 1 in After", True)
        def_set_kitchen = st.checkbox("Set separate kitchen to 1 in After", True)

if mode == "Use scenarios (sliders/toggles)":
    file = st.file_uploader("Upload baseline CSV", type=["csv"])
else:
    file = st.file_uploader("Upload CSV (Before or single file)", type=["csv"])

df_after = None
id_col = None

if mode == "Two CSVs: Before + After":
    file_after = st.file_uploader("Upload After CSV", type=["csv"], key="after_csv")
    if file and file_after:
        df = pd.read_csv(file)
        df2 = pd.read_csv(file_after)
        common = [c for c in df.columns if c in df2.columns]
        id_col = st.selectbox("Select household ID to merge on", options=common, index=0 if common else None)
        if id_col:
            df_after = df2.set_index(id_col)
            df = df.set_index(id_col)
else:
    if file:
        df = pd.read_csv(file)

if file is None:
    st.stop()

# Preview
st.success(f"Loaded data shape: {df.shape}")
with st.expander("Data preview", expanded=False):
    nshow = st.slider("Rows to preview", 5, min(100, len(df)), min(20, len(df)))
    st.dataframe(df.head(nshow))

cols = df.columns.tolist()

# ---------- Column finder ----------
def find(*names, in_after=False):
    source_cols = df_after.columns.tolist() if in_after and (df_after is not None) else cols
    colset = {c.lower(): c for c in source_cols}
    for n in names:
        if n and n.lower() in colset:
            return colset[n.lower()]
    for c in source_cols:
        if any(n.lower() in c.lower() for n in names if n):
            return c
    return None

# Helper to fetch paired before/after columns (for Single CSV mode)
def paired(basename):
    base = None; aft = None
    for c in cols:
        cl = c.lower()
        if cl == basename.lower(): base = c
        if cl == f"{basename.lower()}_after" or cl == f"{basename.lower()} after" or cl == f"after_{basename.lower()}":
            aft = c
    if base is None:
        base = find(basename)
    if aft is None:
        for c in cols:
            if basename.lower() in c.lower() and "after" in c.lower():
                aft = c; break
    return base, aft

# ===== Build indicator scores (0–1) =====
before = df.copy()
weights = {}

house_col, house_col_after = (paired("Residential_Status")[0], None)
if not house_col: house_col = find("House_Type","Housing")
if mode == "Single CSV with After columns":
    tmp,_a = paired(house_col or "Residential_Status")
    if _a: house_col_after = _a

if house_col:
    pos = "Pucca"
    before["House Score"] = (before[house_col].astype(str) == str(pos)).astype(float)
    weights["House Score"] = 0.10

for key, name in [("Electricity","Electricity Score"), ("Separate_Kitchen","Kitchen Score"), ("Toilet","Toilet Score")]:
    base, aft = paired(key) if mode == "Single CSV with After columns" else (find(key), None)
    if mode == "Two CSVs: Before + After" and id_col:
        base = base or key
        aft = base
    if base is not None and base in df.columns:
        before[name] = (pd.to_numeric(df[base], errors="coerce")==1).astype(float)
        weights[name] = {"Electricity Score":0.10, "Kitchen Score":0.10, "Toilet Score":0.15}[name]

land_base = find("Agricultural_Land_sqft","Agri_Land")
if land_base:
    before["Land Binary Score"] = (pd.to_numeric(df[land_base], errors="coerce") > 0).astype(float)
    weights["Land Binary Score"] = 0.05

def add_norm(colname, source_df):
    if not colname or colname not in source_df.columns: 
        return pd.Series(np.nan, index=source_df.index)
    x = pd.to_numeric(source_df[colname], errors="coerce")
    lo, hi = np.nanmin(x), np.nanmax(x)
    return normalize_01(x, lo, hi)

homestead = find("Homestead_Land_sqft","Homestead","Floor_Area")
rooms = find("Number_of_Rooms","Rooms")

before["Homestead Area Score"] = add_norm(homestead, df)
before["Rooms Score"] = add_norm(rooms, df)
weights["Homestead Area Score"] = 0.07; weights["Rooms Score"] = 0.08

occ = find("Primary_Occupation","Occupation")
if occ:
    uniq = before[occ].dropna().astype(str).unique().tolist()
    defaults = {"Jobless":0.2,"Coal Mine":0.55,"Coal Transport":0.6,"Agriculture":0.6,"Self Employed":0.7,"Govt Service":1.0}
    mapping = {u: defaults.get(format_label(u), 0.5) for u in uniq}
    before["Occupation Score"] = before[occ].map(mapping).astype(float)
    weights["Occupation Score"] = 0.20

tot = sum(weights.values()) or 1.0
weights = {k: v/tot for k, v in weights.items()}

after = before.copy()

def use_after_column(col_name_base, fallback_series=None):
    if mode == "Single CSV with After columns":
        b,a = paired(col_name_base)
        if a and a in df.columns:
            ser = df[a]
            return ser
    if mode == "Two CSVs: Before + After" and df_after is not None and id_col is not None:
        if col_name_base in df_after.columns:
            return df_after[col_name_base].reindex(after.index)
        for c in df_after.columns:
            if col_name_base.lower() in c.lower():
                return df_after[c].reindex(after.index)
    return fallback_series

# ===== Descriptive =====
st.header("Descriptive Statistics")
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols]

if num_cols:
    st.subheader("Numeric summary")
    num_summary = df[num_cols].describe().T.reset_index().rename(columns={"index":"Variable"})
    st.dataframe(num_summary)
    make_df_downloads(num_summary, "Numeric Summary", add_png=True)

    st.subheader("Histograms")
    nbins = st.slider("Bins for all histograms", 10, 60, 20, 2)
    for c in num_cols:
        data = df[c].dropna()
        if len(data) == 0: 
            continue
        fig, ax = plt.subplots()
        ax.hist(data, bins=nbins, alpha=0.85)
        ax.set_title(f"Histogram: {format_label(c)}")
        ax.set_xlabel(format_label(c)); ax.set_ylabel("Count")
        st.pyplot(fig)
        make_fig_download(f"Download histogram of {format_label(c)} (PNG)", fig, f"Histogram - {format_label(c)}")

if cat_cols:
    st.subheader("Pie charts")
    default_cats = [c for c in cat_cols if any(k in c.lower() for k in ["caste","occup","residential","building"])]
    selected_cats = st.multiselect("Categorical variables for pie charts", options=cat_cols, default=default_cats or cat_cols[:6])
    for c in selected_cats:
        vc = df[c].astype(str).value_counts(dropna=True).head(12)
        if len(vc) == 0: 
            continue
        fig, ax = plt.subplots()
        ax.pie(vc.values, labels=[format_label(x) for x in vc.index], autopct="%1.1f%%", startangle=90)
        ax.set_title(f"Pie chart: {format_label(c)} (top 12)")
        st.pyplot(fig)
        make_fig_download(f"Download pie of {format_label(c)} (PNG)", fig, f"Pie - {format_label(c)}")

# ===== SIS =====
st.header("SIS (Before vs After)")
# Simulation toggles default from package when no After data present
toggle_elec = st.checkbox("Set Electricity to 1 in After (simulation)", value=enable_pkg and def_set_water)
toggle_kitch = st.checkbox("Set Kitchen to 1 in After (simulation)", value=enable_pkg and def_set_kitchen)
toggle_toilet= st.checkbox("Set Toilet to 1 in After (simulation)", value=enable_pkg and def_set_toilet)

before["SIS Before"] = 0.0
for c,w in weights.items():
    before["SIS Before"] += before.get(c, 0.0) * w

for c in ["Electricity Score","Kitchen Score","Toilet Score"]:
    after_vals = None
    if c == "Electricity Score": after_vals = use_after_column("Electricity")
    if c == "Kitchen Score": after_vals = use_after_column("Separate_Kitchen")
    if c == "Toilet Score": after_vals = use_after_column("Toilet")
    if after_vals is not None:
        after[c] = (pd.to_numeric(after_vals, errors="coerce")==1).astype(float)
    else:
        if c=="Electricity Score" and toggle_elec and c in after.columns: after[c]=1.0
        if c=="Kitchen Score" and toggle_kitch and c in after.columns: after[c]=1.0
        if c=="Toilet Score" and toggle_toilet and c in after.columns: after[c]=1.0

after["SIS After"] = 0.0
for c,w in weights.items():
    after["SIS After"] += after.get(c, 0.0) * w

def label_band(x, b1=0.3, b2=0.6, b3=0.8):
    if np.isnan(x): return "NA"
    if x < b1: return "Low"
    if x < b2: return "Moderate"
    if x < b3: return "High"
    return "Very high"

SIS_b = float(before["SIS Before"].mean())
SIS_a = float(after["SIS After"].mean())
band_b, band_a = label_band(SIS_b), label_band(SIS_a)
delta = SIS_a - SIS_b
if delta > 0.02:
    box = st.success; verdict = "Positive Social Impact"
elif delta < -0.02:
    box = st.error; verdict = "Negative Social Impact"
else:
    box = st.warning; verdict = "Neutral / Inconclusive"
box(f"**Before SIS:** {SIS_b:.3f} ({band_b})  \n**After SIS:** {SIS_a:.3f} ({band_a})  \n**Change:** {delta:+.3f}  \n**Verdict:** {verdict}")

# ===== Health Index (with baseline-bounds access) =====
st.header("Health Index")
st.caption("If *_After columns exist they are used; otherwise defaults apply. Access normalization uses baseline min/max so percentage reductions increase the score.")

def to_bin(series): 
    return (pd.to_numeric(series, errors="coerce")==1).astype(float)

dw_col = find("Drinking_Water","Drinking Water","Water")
toilet_col = find("Toilet","Sanitation")
kitch_col = find("Separate_Kitchen","Kitchen")
dist_col = find("Distance_Hospital_km","Distance Hospital","Hospital_Distance")

sW_b = to_bin(df[dw_col]) if dw_col else pd.Series(np.nan, index=df.index)
sT_b = to_bin(df[toilet_col]) if toilet_col else pd.Series(np.nan, index=df.index)
sK_b = to_bin(df[kitch_col]) if kitch_col else pd.Series(np.nan, index=df.index)

# Access BEFORE (baseline bounds)
if dist_col:
    x_b = pd.to_numeric(df[dist_col], errors="coerce")
    lo_b, hi_b = np.nanmin(x_b), np.nanmax(x_b)
    sA_b = 1.0 - normalize_01(x_b, lo_b, hi_b)
else:
    x_b = pd.Series(np.nan, index=df.index); lo_b = np.nan; hi_b = np.nan
    sA_b = pd.Series(np.nan, index=df.index)

def to_bin_after(base_name, fallback_before, sim_default):
    col_after = use_after_column(base_name)
    if col_after is not None:
        return to_bin(col_after)
    set_to_one = enable_pkg and sim_default
    return pd.Series(1.0 if set_to_one else np.nan, index=df.index).where(~fallback_before.isna(), other=(1.0 if set_to_one else fallback_before))

sW_a = to_bin_after("Drinking_Water", sW_b, def_set_water)
sT_a = to_bin_after("Toilet", sT_b, def_set_toilet)
sK_a = to_bin_after("Separate_Kitchen", sK_b, def_set_kitchen)

dist_after_col = use_after_column(dist_col or "Distance_Hospital_km")
if dist_after_col is not None and dist_col:
    xa = pd.to_numeric(dist_after_col, errors="coerce")
    sA_a = 1.0 - normalize_01(xa, lo_b, hi_b)
else:
    red_default = def_access_red if enable_pkg else 0
    red = st.slider("After: improve hospital access (reduce distance by %)", 0, 100, red_default, 5)
    if dist_col:
        xa = x_b * (1 - red/100.0)
        sA_a = 1.0 - normalize_01(xa, lo_b, hi_b)
    else:
        sA_a = sA_b.copy()

wW = st.number_input("Weight for Drinking Water", 0.0, 1.0, 0.30, 0.05)
wT = st.number_input("Weight for Toilet", 0.0, 1.0, 0.30, 0.05)
wK = st.number_input("Weight for Kitchen", 0.0, 1.0, 0.20, 0.05)
wA = st.number_input("Weight for Access", 0.0, 1.0, 0.20, 0.05)
den = (wW + wT + wK + wA) or 1.0

HI_b = (wW*sW_b.fillna(0) + wT*sT_b.fillna(0) + wK*sK_b.fillna(0) + wA*sA_b.fillna(0)) / den
HI_a = (wW*sW_a.fillna(0) + wT*sT_a.fillna(0) + wK*sK_a.fillna(0) + wA*sA_a.fillna(0)) / den

st.write(f"Mean Health Index — Before: **{np.nanmean(HI_b):.3f}**, After: **{np.nanmean(HI_a):.3f}**")
make_df_downloads(pd.DataFrame({"Health Index Before": HI_b, "Health Index After": HI_a}), "Health Index", add_png=True)

# ===== Asset Index =====
st.header("Asset Index")

lv_before_cols = [c for c in df.columns if c.lower().startswith("livestock_")]
land_bases = [find("Agricultural_Land_sqft"), find("Non_Agricultural_Land_sqft"), find("Homestead_Land_sqft")]
land_before_cols = [c for c in land_bases if c]

def norm_avg(df_src, cols_list):
    if not cols_list:
        return pd.Series(np.nan, index=df_src.index)
    mats = []
    for c in cols_list:
        x = pd.to_numeric(df_src[c], errors="coerce")
        mats.append(normalize_01(x, np.nanmin(x), np.nanmax(x)))
    return pd.DataFrame(mats).T.mean(axis=1)

lv_score_b = norm_avg(df, lv_before_cols)
land_score_b = norm_avg(df, land_before_cols)

if mode == "Single CSV with After columns":
    lv_after_cols = [c for c in df.columns if c.lower().startswith("livestock_") and "after" in c.lower()]
    land_after_cols = []
    for base in ["Agricultural_Land_sqft", "Non_Agricultural_Land_sqft", "Homestead_Land_sqft"]:
        _, a = paired(base)
        if a: land_after_cols.append(a)
elif mode == "Two CSVs: Before + After" and df_after is not None:
    lv_after_cols = [c for c in df_after.columns if c.lower().startswith("livestock_")]
    land_after_cols = [c for c in df_after.columns if "land" in c.lower() or "homestead" in c.lower()]
else:
    lv_after_cols = []
    land_after_cols = []

if lv_after_cols or land_after_cols:
    lv_score_a = norm_avg(df if mode!="Two CSVs: Before + After" else df_after, lv_after_cols) if lv_after_cols else lv_score_b.copy()
    land_score_a = norm_avg(df if mode!="Two CSVs: Before + After" else df_after, land_after_cols) if land_after_cols else land_score_b.copy()
else:
    st.caption("No explicit after asset columns found — using scenario multipliers")
    multL_default = def_livestock_mult if enable_pkg else 1.0
    multA_default = def_land_mult if enable_pkg else 1.0
    multL = st.number_input("After multiplier: Livestock score", 0.0, 5.0, multL_default, 0.05)
    multA = st.number_input("After multiplier: Land score", 0.0, 5.0, multA_default, 0.05)
    lv_score_a = np.clip(lv_score_b * multL, 0, 1)
    land_score_a = np.clip(land_score_b * multA, 0, 1)

wL = st.number_input("Weight: Livestock", 0.0, 1.0, 0.5, 0.05)
wLand = st.number_input("Weight: Land", 0.0, 1.0, 0.5, 0.05)
sw = (wL + wLand) or 1.0
AI_b = (wL/sw)*lv_score_b + (wLand/sw)*land_score_b
AI_a = (wL/sw)*lv_score_a + (wLand/sw)*land_score_a

st.write(f"Mean Asset Index — Before: **{np.nanmean(AI_b):.3f}**, After: **{np.nanmean(AI_a):.3f}**")
make_df_downloads(pd.DataFrame({"Asset Index Before": AI_b, "Asset Index After": AI_a}), "Asset Index", add_png=True)

# ===== Gini & Lorenz =====
st.header("Inequality (Gini) — Income / Expenditure / Savings")
st.caption("Note: a pure economy wide multiplier does not change Gini (scale-invariant). Use bottom 40 percent extra, or supply actual After columns.")

def series_from_after_or_sim(basename, friendly, def_g=0, def_b40=0):
    base_col = find(basename)
    after_col = use_after_column(base_col or basename)
    series_b = pd.to_numeric(df[base_col], errors="coerce") if base_col in df.columns else pd.Series(dtype=float, index=df.index)
    if after_col is not None:
        series_a = pd.to_numeric(after_col, errors="coerce")
        st.success(f"Using After column for {friendly}")
        return series_b, series_a
    with st.expander(f"{friendly} — Scenario", expanded=(friendly=="Income")):
        pct_global = st.slider(f"Economy wide change (%) — {friendly}", -100, 500, def_g, 5, key=f"{friendly}_g")
        pct_b40 = st.slider(f"Extra change for poorest 40% (%) — {friendly}", -100, 500, def_b40, 5, key=f"{friendly}_b40")
    ser = series_b.copy()
    glob_mult = 1.0 + pct_global/100.0
    b40_mult = 1.0 + pct_b40/100.0
    order = np.argsort(ser.fillna(ser.max()+1e9).values)
    cut = int(np.ceil(0.4*len(ser)))
    idx_bottom = ser.index[order[:cut]]
    series_a = ser * glob_mult
    series_a.loc[idx_bottom] = series_a.loc[idx_bottom] * b40_mult
    return series_b, series_a

def_g_inc = def_inc_global if (enable_pkg) else 0
def_b_inc = def_inc_b40 if (enable_pkg) else 0
def_g_sav = def_sav_global if (enable_pkg) else 0
def_b_sav = def_sav_b40 if (enable_pkg) else 0

inc_b, inc_a = series_from_after_or_sim("Monthly_Income", "Income", def_g_inc, def_b_inc)
exp_b, exp_a = series_from_after_or_sim("Monthly_Expenditure", "Expenditure", 0, 0)
sav_b, sav_a = series_from_after_or_sim("Monthly_Savings", "Savings", def_g_sav, def_b_sav)

st.write(f"Income Gini — Before: **{gini_coefficient(inc_b):.3f}**, After: **{gini_coefficient(inc_a):.3f}**")
st.write(f"Expenditure Gini — Before: **{gini_coefficient(exp_b):.3f}**, After: **{gini_coefficient(exp_a):.3f}**")
st.write(f"Savings Gini — Before: **{gini_coefficient(sav_b):.3f}**, After: **{gini_coefficient(sav_a):.3f}**")

st.subheader("Lorenz curve (Before vs After) with Gini overlay")
metric = st.radio("Metric to plot", options=["Income","Expenditure","Savings"], index=0, horizontal=True)

def plot_lorenz(before_vals, after_vals, title):
    x_b, y_b = lorenz_points(before_vals); x_a, y_a = lorenz_points(after_vals)
    fig, ax = plt.subplots()
    ax.plot([0,1], [0,1], linestyle="--", linewidth=1)
    ax.plot(x_b, y_b, label=f"Before (Gini {gini_coefficient(before_vals):.3f})")
    ax.plot(x_a, y_a, label=f"After (Gini {gini_coefficient(after_vals):.3f})")
    ax.set_xlabel("Cumulative share of households")
    ax.set_ylabel("Cumulative share of value")
    ax.set_title(format_label(title))
    ax.legend()
    return fig

if metric == "Income":
    fig = plot_lorenz(inc_b, inc_a, "Lorenz Curve — Income")
elif metric == "Expenditure":
    fig = plot_lorenz(exp_b, exp_a, "Lorenz Curve — Expenditure")
else:
    fig = plot_lorenz(sav_b, sav_a, "Lorenz Curve — Savings")
st.pyplot(fig)
make_fig_download("Download Lorenz curve (PNG)", fig, f"Lorenz Curve - {metric}")

# ===== Proportion tests =====
st.header("One sided Proportion Tests (Overall)")
bin_cols = []
for c in before.columns:
    if c.endswith("Score"):
        vals = pd.to_numeric(before[c], errors="coerce").dropna().unique()
        if set(vals).issubset({0.0,1.0}):
            bin_cols.append(c)

alpha_overall = st.number_input("Significance level (alpha) for overall tests", 0.001, 0.2, 0.05, 0.005)
rows = []
n1, n2 = len(before), len(after)
for c in bin_cols:
    x1 = int(pd.to_numeric(before[c], errors="coerce").fillna(0).sum())
    x2 = int(pd.to_numeric(after[c], errors="coerce").fillna(0).sum())
    p1 = x1/n1 if n1>0 else np.nan
    p2 = x2/n2 if n2>0 else np.nan
    p = (x1+x2)/(n1+n2); se = np.sqrt(p*(1-p)*(1/n1 + 1/n2))
    if se == 0:
        z = np.nan; pval = np.nan
    else:
        z = (p1 - p2)/se
        from math import erf
        pval = 0.5*(1+erf(z/np.sqrt(2)))
    decision = "Reject H0 (p before < p after)" if (not np.isnan(pval) and pval < alpha_overall) else "Fail to reject H0"
    rows.append({"Indicator": format_label(c), "p before": p1, "p after": p2, "z": z, "p value": pval, "Decision": decision})
overall_results = pd.DataFrame(rows).sort_values("p value")
st.dataframe(overall_results)
make_df_downloads(overall_results, "Overall Proportion Tests", add_png=True)

# ===== SIS Summary =====
st.header("SIS Summary")
st.write(f"Mean SIS — Before: **{SIS_b:.3f}**, After: **{SIS_a:.3f}**, Δ = **{delta:.3f}**")

# ===== Export combined scored data =====
scored = pd.DataFrame(index=df.index)
scored["SIS Before"] = before["SIS Before"]; scored["SIS After"] = after["SIS After"]
scored["Health Index Before"] = HI_b; scored["Health Index After"] = HI_a
scored["Asset Index Before"] = AI_b; scored["Asset Index After"] = AI_a
scored["Income Before"] = inc_b; scored["Income After"] = inc_a
scored["Expenditure Before"] = exp_b; scored["Expenditure After"] = exp_a
scored["Savings Before"] = sav_b; scored["Savings After"] = sav_a
st.subheader("Download results")
make_df_downloads(scored.reset_index(), "Scored Data (All metrics)", add_png=False)
