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

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Aadhar Nexus | AI Command Center",
    layout="wide",
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

# --- 2. ETHEREAL STYLING ---
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #ffffff;
    }
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 15px;
        backdrop-filter: blur(5px);
    }
    h1, h2, h3 { color: #ffffff; font-weight: 200; }
</style>
""", unsafe_allow_html=True)

# --- 3. SIDEBAR & FILE UPLOADER ---
st.sidebar.markdown("## 🧬 Aadhar Nexus")
st.sidebar.info("Upload Biometric CSVs to initialize.")
uploaded_files = st.sidebar.file_uploader("📂 Upload Biometric CSVs", accept_multiple_files=True, type="csv")

# --- 4. DATA LOADER ---
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
        df.columns = [c.lower() for c in df.columns]
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], dayfirst=True)
            
            # Create total_bio if missing
            if 'total_bio' not in df.columns:
                if 'bio_age_5_17' in df.columns and 'bio_age_17_' in df.columns:
                    df['total_bio'] = df['bio_age_5_17'] + df['bio_age_17_']
                else:
                    return None, "invalid_columns"
            return df, "success"
    return None, "empty"

df, status = load_data(uploaded_files)

# --- 5. VALIDATION ---
if status != "success":
    st.title("🧬 Aadhar Operations Command")
    if status == "missing":
        st.warning("⚠️ Waiting for Data Uplink...")
        st.info("Please upload `api_data_aadhar_biometric_....csv` files in the sidebar.")
    elif status == "invalid_columns":
        st.error("Error: CSV files missing required columns (bio_age_5_17, bio_age_17_).")
    st.stop()

# --- 6. FILTERS ---
min_date, max_date = df['date'].min(), df['date'].max()
st.sidebar.markdown("### 🗓️ Time Travel")
date_range = st.sidebar.date_input("Filter Range", [min_date, max_date])

# Robust Date Filtering
start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1]) if len(date_range) > 1 else start_date
mask = (df['date'] >= start_date) & (df['date'] <= end_date)
filtered_df = df.loc[mask]

if filtered_df.empty:
    st.error("No data available for the selected date range.")
    st.stop()

# --- 7. DASHBOARD (Fixed KPIs) ---
st.markdown("# 🧬 Aadhar Operations Command")
st.markdown(f"### *System Status: Active | Monitoring {filtered_df['state'].nunique()} States*")

col1, col2, col3, col4 = st.columns(4)

# FIX: Calculate Daily Aggregates FIRST
daily_sums = filtered_df.groupby('date')['total_bio'].sum()

total_req = filtered_df['total_bio'].sum()
peak_day_load = daily_sums.max() if not daily_sums.empty else 0
avg_load = daily_sums.mean() if not daily_sums.empty else 0
top_state = filtered_df.groupby('state')['total_bio'].sum().idxmax() if not filtered_df.empty else "N/A"

col1.metric("Total Transactions", f"{total_req:,.0f}", "Live")
col2.metric("Peak System Load", f"{peak_day_load:,.0f}", "Critical Threshold")
col3.metric("Avg Daily Traffic", f"{avg_load:,.0f}", "Normal")
col4.metric("Highest Activity Zone", str(top_state), "Hotspot")

st.markdown("---")

# --- 8. TABS ---
tab_eda, tab_ai, tab_sim, tab_pred = st.tabs(["📊 Analytics", "🧠 AI Clusters", "📉 Load Balancer", "🔮 Forecast"])

# TAB 1: Analytics
with tab_eda:
    c1, c2 = st.columns([2, 1])
    with c1:
        daily_df = daily_sums.reset_index()
        daily_df['7_Day_Avg'] = daily_df['total_bio'].rolling(7).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_df['date'], y=daily_df['total_bio'], name='Raw', line=dict(color='#00c6ff')))
        fig.add_trace(go.Scatter(x=daily_df['date'], y=daily_df['7_Day_Avg'], name='Trend', line=dict(color='#ff00cc')))
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        top_states = filtered_df.groupby('state')['total_bio'].sum().nlargest(10).reset_index()
        fig2 = px.bar(top_states, x='total_bio', y='state', orientation='h', title="Top States", color='total_bio')
        fig2.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

# TAB 2: AI
with tab_ai:
    st.subheader("🧠 District Risk Clustering")
    dist_stats = filtered_df.groupby(['state', 'district'])[['total_bio', 'bio_age_5_17']].sum().reset_index()
    
    if len(dist_stats) > 3:
        scaler = StandardScaler()
        X = scaler.fit_transform(dist_stats[['total_bio', 'bio_age_5_17']])
        kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
        dist_stats['Cluster'] = kmeans.labels_
        
        # Sort clusters by load
        cluster_map = {c: 'High' if m == dist_stats.groupby('Cluster')['total_bio'].mean().max() else 'Low' 
                       for c, m in dist_stats.groupby('Cluster')['total_bio'].mean().items()}
        # Simple mapping fallback
        mean_loads = dist_stats.groupby('Cluster')['total_bio'].mean().sort_values()
        risk_map = {mean_loads.index[0]: 'Safe', mean_loads.index[1]: 'Medium', mean_loads.index[2]: 'Critical'}
        dist_stats['Risk'] = dist_stats['Cluster'].map(risk_map)
        
        fig_ai = px.scatter(dist_stats, x='bio_age_5_17', y='total_bio', color='Risk', hover_data=['district'], 
                            color_discrete_map={'Critical': '#ff0055', 'Medium': '#00c6ff', 'Safe': '#00ff99'})
        fig_ai.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_ai, use_container_width=True)
    else:
        st.info("Not enough data points for AI clustering.")

# TAB 3: Simulation
with tab_
