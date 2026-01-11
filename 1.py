import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings for a clean UI
warnings.filterwarnings('ignore')

# --- 1. PAGE CONFIGURATION (The Foundation) ---
st.set_page_config(
    page_title="Aadhar Nexus | AI Command Center",
    layout="wide",
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

# --- 2. ETHEREAL STYLING (Custom CSS) ---
st.markdown("""
<style>
    /* Main Background - Deep Futuristic Blue/Black Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #ffffff;
    }
    
    /* Glassmorphism Cards for Metrics */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 200;
        text-shadow: 0 0 10px rgba(0, 198, 255, 0.5);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255,255,255,0.05);
        border-radius: 10px;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00c6ff;
        color: black;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. SIDEBAR & FILE UPLOADER ---
st.sidebar.markdown("## 🧬 Aadhar Nexus")
st.sidebar.markdown("---")

st.sidebar.info("Upload your Biometric Data CSVs below to initialize the Command Center.")
uploaded_files = st.sidebar.file_uploader("📂 Upload Biometric CSVs", accept_multiple_files=True, type="csv")

# --- 4. INTELLIGENT DATA LOADER ---
@st.cache_data
def load_data(files):
    if not files:
        return None, "missing"
    
    df_list = []
    for f in files:
        try:
            temp = pd.read_csv(f)
            df_list.append(temp)
        except:
            pass
            
    if df_list:
        df = pd.concat(df_list, ignore_index=True)
        # Ensure column names are lowercase
        df.columns = [c.lower() for c in df.columns]
        
        # Date Parsing
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], dayfirst=True)
            
            # Create Total Bio column if it doesn't exist
            if 'total_bio' not in df.columns:
                if 'bio_age_5_17' in df.columns and 'bio_age_17_' in df.columns:
                    df['total_bio'] = df['bio_age_5_17'] + df['bio_age_17_']
                else:
                    return None, "invalid_columns"
                    
            return df, "success"
    return None, "empty"

# Load Data based on User Uploads
df, status = load_data(uploaded_files)

# --- 5. DATA VALIDATION ---
if status == "missing":
    st.markdown("# 🧬 Aadhar Operations Command")
    st.warning("⚠️ Waiting for Data Uplink...")
    st.info("Please drag and drop the `api_data_aadhar_biometric_....csv` files in the sidebar.")
    st.stop()
elif status == "invalid_columns":
    st.error("Error: The uploaded files do not have the required Biometric columns (`bio_age_5_17`, `bio_age_17_`).")
    st.stop()

# --- 6. DATE FILTERS ---
# Date Filter in Sidebar
min_date = df['date'].min()
max_date = df['date'].max()
st.sidebar.markdown("### 🗓️ Time Travel")
date_range = st.sidebar.date_input(
    "Filter Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Apply Filter
mask = (df['date'].dt.date >= date_range[0]) & (df['date'].dt.date <= date_range[1])
filtered_df = df.loc[mask]

# --- 7. HEADS UP DISPLAY (KPIs) ---
st.markdown("# 🧬 Aadhar Operations Command")
st.markdown(f"### *System Status: Active | Monitoring {filtered_df['state'].nunique()} States*")
st.write("")

# KPIs
col1, col2, col3, col4 = st.columns(4)
total_req = filtered_df['total_bio'].sum()
peak_day_load = filtered_df.groupby('date')['total_bio'].sum().max()
avg_load = filtered_df.groupby('date')['total_bio'].mean()
top_state = filtered_df.groupby('state')['total_bio'].sum().idxmax()

col1.metric("Total Transactions", f"{total_req:,.0f}", "Live")
col2.metric("Peak System Load", f"{peak_day_load:,.0f}", "Critical Threshold")
col3.metric("Avg Daily Traffic", f"{avg_load:,.0f}", "Normal")
col4.metric("Highest Activity Zone", top_state, "Hotspot")

st.markdown("---")

# --- 8. MAIN INTERFACE TABS ---
tab_eda, tab_ai, tab_sim, tab_pred = st.tabs([
    "📊 Deep Dive Analytics", 
    "🧠 AI Cluster Intelligence", 
    "📉 Smart Load Balancer", 
    "🔮 Future Prophet"
])

# === TAB 1: VISUAL ANALYTICS (Sunburst & Time Series) ===
with tab_eda:
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        st.subheader("Interactive Temporal Flow")
        # Aggregating for time series
        daily = filtered_df.groupby('date')['total_bio'].sum().reset_index()
        # Add Moving Average for "Smoothness"
        daily['7_Day_Avg'] = daily['total_bio'].rolling(window=7).mean()
        
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=daily['date'], y=daily['total_bio'], mode='lines', name='Raw Traffic', line=dict(color='#00c6ff', width=1)))
        fig_line.add_trace(go.Scatter(x=daily['date'], y=daily['7_Day_Avg'], mode='lines', name='Trend (7-Day Avg)', line=dict(color='#ff00cc', width=3)))
        fig_line.update_layout(
            template="plotly_dark", 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode="x unified"
        )
        st.plotly_chart(fig_line, use_container_width=True)
        
    with col_b:
        st.subheader("Geographic Hierarchy")
        st.markdown("Click to drill down: State -> District")
        # Sunburst Chart
        hier_df = filtered_df.groupby(['state', 'district'])['total_bio'].sum().reset_index()
        # Limit to top states for rendering speed
        top_states = hier_df.groupby('state')['total_bio'].sum().nlargest(10).index
        hier_df_small = hier_df[hier_df['state'].isin(top_states)]
        
        fig_sun = px.sunburst(
            hier_df_small, 
            path=['state', 'district'], 
            values='total_bio',
            color='total_bio',
            color_continuous_scale='Bluyl'
        )
        fig_sun.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_sun, use_container_width=True)

# === TAB 2: AI CLUSTERING (Unsupervised Learning) ===
with tab_ai:
    st.subheader("🧠 Unsupervised Anomaly Detection")
    st.write("The AI has analyzed district-level patterns to group them into 'Behavior Clusters'.")
    
    # Prepare Data for Clustering
    district_stats = filtered_df.groupby(['state', 'district']).agg({
        'total_bio': 'sum',
        'bio_age_5_17': 'sum',
        'bio_age_17_': 'sum'
    }).reset_index()
    
    # Machine Learning Logic
    X = district_stats[['total_bio', 'bio_age_5_17', 'bio_age_17_']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    district_stats['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Rename clusters based on size (Logic: High mean = High Traffic)
    cluster_means = district_stats.groupby('Cluster')['total_bio'].mean().sort_values()
    cluster_map = {
        cluster_means.index[0]: 'Low Volume (Safe)',
        cluster_means.index[1]: 'Medium Volume (Stable)',
        cluster_means.index[2]: 'High Volume (Critical Risk)'
    }
    district_stats['Risk Profile'] = district_stats['Cluster'].map(cluster_map)
    
    # Scatter Plot
    fig_cluster = px.scatter(
        district_stats, 
        x='bio_age_5_17', 
        y='bio_age_17_', 
        color='Risk Profile',
        size='total_bio',
        hover_data=['state', 'district'],
        title="District Risk Clustering (AI Segmented)",
        color_discrete_map={'High Volume (Critical Risk)': '#ff0055', 'Medium Volume (Stable)': '#00c6ff', 'Low Volume (Safe)': '#00ff99'}
    )
    fig_cluster.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=600)
    st.plotly_chart(fig_cluster, use_container_width=True)
    
    # Show High Risk List
    st.warning("⚠️ High Risk Districts Identified by AI:")
    st.dataframe(district_stats[district_stats['Risk Profile'] == 'High Volume (Critical Risk)'][['state', 'district', 'total_bio']].sort_values('total_bio', ascending=False).head(10), hide_index=True)

# === TAB 3: LOAD BALANCER (The Simulation) ===
with tab_sim:
    st.subheader("📉 Intelligent Traffic Smoothing Simulator")
    
    # Simulation Layout
    c1, c2 = st.columns([1, 2])
    
    daily_agg = filtered_df.groupby('date')['total_bio'].sum().reset_index().sort_values('date')
    current_peak = daily_agg['total_bio'].max()
    safe_cap = daily_agg['total_bio'].mean() * 1.5
    
    with c1:
        st.markdown("### Controls")
        capacity = st.slider(
            "Define Server Capacity (Daily Requests)",
            min_value=100000,
            max_value=int(current_peak),
            value=int(safe_cap),
            step=50000
        )
        st.info("Adjust the slider to see how spreading traffic prevents crashes.")

    with c2:
        # Optimization Algorithm
        sim_data = daily_agg.copy()
        queue = 0
        optimized = []
        
        for load in sim_data['total_bio']:
            total = load + queue
            if total > capacity:
                processed = capacity
                queue = total - capacity
            else:
                processed = total
                queue = 0
            optimized.append(processed)
            
        sim_data['Managed Load'] = optimized
        
        # Plot
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(x=sim_data['date'], y=sim_data['total_bio'], fill='tozeroy', name='Original Crash Risk', line=dict(color='#ff4b4b')))
        fig_sim.add_trace(go.Scatter(x=sim_data['date'], y=sim_data['Managed Load'], fill='tonexty', name='Optimized Flow', line=dict(color='#00ff99')))
        fig_sim.add_hline(y=capacity, line_dash="dash", annotation_text="Max Capacity", line_color="white")
        fig_sim.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_sim, use_container_width=True)

# === TAB 4: PREDICTION (Forecasting) ===
with tab_pred:
    st.subheader("🔮 Predictive Intelligence")
    
    # Feature Engineering
    df_model = daily_agg.copy()
    df_model['day'] = df_model['date'].dt.day
    df_model['month'] = df_model['date'].dt.month
    df_model['dow'] = df_model['date'].dt.dayofweek
    df_model['is_start'] = (df_model['day'] == 1).astype(int)
    
    X = df_model[['day', 'month', 'dow', 'is_start']]
    y = df_model['total_bio']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    col_input, col_result = st.columns(2)
    
    with col_input:
        pred_date = st.date_input("Forecast Date", value=pd.to_datetime("2026-01-01"))
        run_pred = st.button("Generate Forecast", type="primary")
        
    with col_result:
        if run_pred:
            # Predict
            feats = [[pred_date.day, pred_date.month, pred_date.weekday(), 1 if pred_date.day == 1 else 0]]
            prediction = model.predict(feats)[0]
            
            st.metric("Expected Load", f"{int(prediction):,}")
            
            if prediction > capacity:
                st.error("⚠️ Overload Predicted. Activate Load Balancer.")
                st.progress(100)
            else:
                st.success("✅ Within Operational Limits.")
                st.progress(int((prediction/capacity)*100))

# --- FOOTER ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>Developed for Next-Gen Governance | Powered by Python </div>", unsafe_allow_html=True)
