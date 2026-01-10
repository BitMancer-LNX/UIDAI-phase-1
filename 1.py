import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

# Set page configuration
st.set_page_config(page_title="Aadhar Service Analytics Dashboard", layout="wide")

st.title("🇮🇳 Aadhar Service Analytics & Prediction Dashboard")
st.markdown("""
This dashboard analyzes Biometric, Demographic, and Enrolment trends to identify **anomalies** and **forecast service load**.
""")

# --- Sidebar: File Uploads ---
st.sidebar.header("📂 Data Upload")
st.sidebar.info("Upload the CSV files for each category below.")

bio_files = st.sidebar.file_uploader("Upload Biometric CSVs", accept_multiple_files=True, type="csv")
demo_files = st.sidebar.file_uploader("Upload Demographic CSVs", accept_multiple_files=True, type="csv")
enrol_files = st.sidebar.file_uploader("Upload Enrolment CSVs", accept_multiple_files=True, type="csv")

# --- Helper Function to Load Data ---
@st.cache_data
def load_data(files):
    if not files:
        return None
    df_list = []
    for file in files:
        df_list.append(pd.read_csv(file))
    return pd.concat(df_list, ignore_index=True)

# --- Main App Logic ---
if bio_files and demo_files and enrol_files:
    with st.spinner('Loading and Processing Data...'):
        # 1. Load Data
        df_bio = load_data(bio_files)
        df_demo = load_data(demo_files)
        df_enrol = load_data(enrol_files)

        # 2. Preprocessing
        # Date Conversion
        df_bio['date'] = pd.to_datetime(df_bio['date'], dayfirst=True)
        df_demo['date'] = pd.to_datetime(df_demo['date'], dayfirst=True)
        df_enrol['date'] = pd.to_datetime(df_enrol['date'], dayfirst=True)

        # Calculate Totals
        df_bio['total_bio'] = df_bio['bio_age_5_17'] + df_bio['bio_age_17_']
        df_demo['total_demo'] = df_demo['demo_age_5_17'] + df_demo['demo_age_17_']
        df_enrol['total_enrol'] = df_enrol['age_0_5'] + df_enrol['age_5_17'] + df_enrol['age_18_greater']

    st.success("Data Successfully Loaded!")

    # --- TABBED VIEW ---
    tab1, tab2, tab3 = st.tabs(["📊 EDA & Trends", "⚠️ Anomaly Detection", "🔮 Predictive Model"])

    # === TAB 1: EDA ===
    with tab1:
        st.header("Service Usage Trends")
        
        # Key Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Biometric Updates", f"{df_bio['total_bio'].sum():,}")
        col2.metric("Total Demographic Updates", f"{df_demo['total_demo'].sum():,}")
        col3.metric("Total New Enrolments", f"{df_enrol['total_enrol'].sum():,}")

        # Plot 1: Activity by State
        st.subheader("Activity Volume by State")
        state_bio = df_bio.groupby('state')['total_bio'].sum().sort_values(ascending=False).head(10).reset_index()
        fig_state = px.bar(state_bio, x='state', y='total_bio', title="Top 10 States - Biometric Activity", color='total_bio')
        st.plotly_chart(fig_state, use_container_width=True)

        # Plot 2: Time Series
        st.subheader("Daily Transaction Volume")
        daily_bio = df_bio.groupby('date')['total_bio'].sum().reset_index()
        fig_time = px.line(daily_bio, x='date', y='total_bio', title="Daily Biometric Transactions (Notice the Spikes)")
        st.plotly_chart(fig_time, use_container_width=True)

    # === TAB 2: ANOMALIES ===
    with tab2:
        st.header("Anomaly Detection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. The 'First of Month' Surge")
            st.markdown("We detected massive spikes on the 1st of every month.")
            
            # Highlight Spikes
            daily_bio['day'] = daily_bio['date'].dt.day
            spikes = daily_bio[daily_bio['day'] == 1].sort_values('total_bio', ascending=False).head(5)
            st.write("Top Spike Dates:")
            st.dataframe(spikes[['date', 'total_bio']].style.format({"total_bio": "{:,}"}))

        with col2:
            st.subheader("2. Unusual Activity Ratios")
            st.markdown("Regions with extremely high Biometric Updates vs. New Enrolments.")
            
            # Ratio Calculation
            s_bio = df_bio.groupby('state')['total_bio'].sum()
            s_enrol = df_enrol.groupby('state')['total_enrol'].sum()
            ratio_df = pd.DataFrame({'Biometric': s_bio, 'Enrolment': s_enrol})
            ratio_df['Ratio'] = ratio_df['Biometric'] / ratio_df['Enrolment']
            
            outliers = ratio_df.sort_values('Ratio', ascending=False).head(10)
            st.dataframe(outliers.style.format("{:.2f}"))

    # === TAB 3: PREDICTION ===
    with tab3:
        st.header("Service Load Forecaster")
        st.markdown("This model predicts the expected load to help with server scaling.")

        # Prepare Data for Training
        daily_data = df_bio.groupby('date')['total_bio'].sum().reset_index()
        daily_data['day_of_week'] = daily_data['date'].dt.dayofweek
        daily_data['day_of_month'] = daily_data['date'].dt.day
        daily_data['month'] = daily_data['date'].dt.month
        daily_data['is_start_of_month'] = (daily_data['day_of_month'] == 1).astype(int)

        X = daily_data[['day_of_week', 'day_of_month', 'month', 'is_start_of_month']]
        y = daily_data['total_bio']

        # Train Model
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)

        # User Input
        st.subheader("Check Forecast for a Future Date")
        input_date = st.date_input("Select Date")
        
        if st.button("Predict Load"):
            # Create feature vector
            input_features = pd.DataFrame({
                'day_of_week': [input_date.weekday()],
                'day_of_month': [input_date.day],
                'month': [input_date.month],
                'is_start_of_month': [1 if input_date.day == 1 else 0]
            })

            prediction = model.predict(input_features)[0]

            # Display Result
            st.metric(label=f"Predicted Transactions for {input_date}", value=f"{int(prediction):,}")
            
            if prediction > 5_000_000:
                st.error("🚨 CRITICAL LOAD PREDICTED: Ensure Max Server Capacity!")
            elif prediction > 1_000_000:
                st.warning("⚠️ HIGH LOAD: Monitor Systems.")
            else:
                st.success("✅ NORMAL LOAD: Standard Operations.")

else:
    st.info("👋 Please upload the CSV files in the sidebar to begin analysis.")
