import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# --- Configuration ---
st.set_page_config(
    page_title="EVOLUTION: EV Adoption Forecast ⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a more attractive, creative, and dashing look ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&family=Outfit:wght@400;700;900&family=Space+Mono:wght@400;700&display=swap');

    body {
        font-family: 'Poppins', sans-serif;
        background-color: #0d1117; /* Dark GitHub-like background */
        color: #e6edf3;
    }

    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }

    /* Target specific Streamlit containers for consistent background */
    .css-1d391kg, .css-1siy2j7, .st-emotion-cache-z5fcl4 { /* Main content and sidebar backgrounds */
        background-color: #0d1117;
    }

    /* Header Animation and Styling */
    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes borderGlow {
        0% { box-shadow: 0px 0px 15px rgba(250, 204, 21, 0.4); }
        50% { box-shadow: 0px 0px 30px rgba(250, 204, 21, 0.9); }
        100% { box-shadow: 0px 0px 15px rgba(250, 204, 21, 0.4); }
    }
    .header-container {
        background: linear-gradient(135deg, #0f172a, #1e293b, #0f172a);
        background-size: 300% 300%;
        animation: gradientMove 12s ease infinite;
        padding: 30px; /* Increased padding */
        border-radius: 15px;
        border: 3px solid #facc15;
        text-align: center;
        animation: borderGlow 3s ease-in-out infinite;
        margin-bottom: 40px; /* Increased margin */
        position: relative;
        overflow: hidden;
    }
    .header-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at center, rgba(250, 204, 21, 0.1) 0%, transparent 70%);
        animation: rotateLight 10s linear infinite;
        opacity: 0.7;
    }
    @keyframes rotateLight {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .header-title {
        font-family: 'Outfit', sans-serif;
        font-size: 52px;
        font-weight: 700;
        color: #facc15;
        text-shadow: 0px 0px 25px rgba(250, 204, 21, 1.0);
        margin-bottom: 10px;
        letter-spacing: 4px;
    }
    .header-subtitle {
        font-family: 'Poppins', sans-serif;
        font-size: 26px;
        color: #ffffff;
        margin-top: 5px;
        font-weight: 300;
        text-shadow: 0px 0px 12px rgba(255, 255, 255, 0.4);
    }

    /* Welcome Banner */
    .welcome-banner {
        text-align: center;
        font-size: 28px;
        font-weight: 700;
        color: #facc15;
        background: linear-gradient(90deg, rgba(30,41,59,0.95), rgba(51,65,85,0.98));
        padding: 20px 25px;
        margin: 20px auto 40px;
        border-radius: 15px;
        border: 3px solid #facc15;
        width: 85%;
        box-shadow: 0 0 20px rgba(250, 204, 21, 0.5);
        text-shadow: 0px 0px 10px rgba(250, 204, 21, 0.9);
        transition: transform 0.4s ease-in-out, box-shadow 0.4s ease-in-out;
    }
    .welcome-banner:hover {
        transform: translateY(-8px) scale(1.005);
        box-shadow: 0 0 35px rgba(250, 204, 21, 0.8);
    }
    .welcome-banner span {
        font-family: 'Outfit', sans-serif;
        font-weight: 900;
    }
    .welcome-banner .sub-text {
        font-size: 20px;
        color: #e6edf3;
        font-weight: 400;
    }

    /* Info Cards (if you were to add them) */
    .info-card {
        background: linear-gradient(145deg, #1a222c, #0d1117);
        padding: 25px;
        border-radius: 12px;
        border: 2px solid #2d333b;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
        margin-bottom: 25px;
        transition: all 0.3s ease-in-out;
    }
    .info-card:hover {
        border-color: #38bdf8;
        box-shadow: 0 8px 16px rgba(56, 189, 248, 0.3);
        transform: translateY(-5px);
    }
    .info-card h3 {
        color: #facc15;
        font-size: 22px;
        margin-bottom: 12px;
        font-family: 'Outfit', sans-serif;
        text-shadow: 0px 0px 8px rgba(250, 204, 21, 0.6);
    }
    .info-card p {
        color: #e6edf3;
        font-size: 17px;
    }
    .info-card .metric {
        font-size: 36px;
        font-weight: bold;
        color: #38bdf8;
        font-family: 'Space Mono', monospace;
        text-shadow: 0px 0px 10px rgba(56, 189, 248, 0.8);
    }

    /* Growth Insight Block */
    .growth-insight-block {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        border: 3px solid #facc15;
        margin-top: 30px;
        box-shadow: 0px 0px 20px rgba(250, 204, 21, 0.6);
        transition: all 0.4s ease-in-out;
    }
    .growth-insight-block:hover {
        box-shadow: 0px 0px 40px rgba(250, 204, 21, 1.0);
        transform: scale(1.02);
    }
    .growth-insight-block h2 {
        color:#facc15;
        margin-bottom:15px;
        font-family: 'Outfit', sans-serif;
        font-size: 38px;
        text-shadow: 0px 0px 15px rgba(250, 204, 21, 0.9);
    }
    .growth-insight-block p {
        font-size: 32px;
        color:white;
        margin: 10px 0;
        font-family: 'Space Mono', monospace;
    }
    .growth-insight-block .subtitle {
        font-size: 18px;
        color:#94a3b8;
        font-family: 'Poppins', sans-serif;
    }

    /* Comparison Block */
    .comparison-block {
        background: linear-gradient(145deg, #0f172a, #1e293b);
        padding: 35px;
        border-radius: 20px;
        text-align: center;
        border: 4px solid #facc15;
        margin-top: 40px;
        box-shadow: 0px 0px 25px rgba(250, 204, 21, 0.7);
        transition: all 0.4s ease-in-out;
    }
    .comparison-block:hover {
        box-shadow: 0px 0px 45px rgba(250, 204, 21, 1.0);
        transform: translateY(-7px) scale(1.005);
    }
    .comparison-block h2 {
        color:#facc15;
        font-size: 38px;
        font-weight: bold;
        margin-bottom: 25px;
        text-shadow: 0px 0px 18px rgba(250, 204, 21, 1.0);
        font-family: 'Outfit', sans-serif;
    }
    .comparison-block p {
        font-size: 22px;
        color:white;
        margin: 0;
        line-height: 1.8;
        font-family: 'Poppins', sans-serif;
    }

    /* Footer */
    .footer-container {
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #0f172a, #1e293b);
        border-radius: 12px;
        border: 3px solid #facc15;
        box-shadow: 0px 0px 15px rgba(250, 204, 21, 0.6);
        margin-top: 40px;
    }
    .footer-container p {
        color: #facc15;
        font-size: 20px;
        font-weight: bold;
        margin: 0;
        font-family: 'Poppins', sans-serif;
    }
    .footer-container a {
        color: #58a6ff;
        font-size: 18px;
        text-decoration: none;
        font-weight: bold;
        transition: color 0.3s ease-in-out, text-shadow 0.3s ease-in-out;
        font-family: 'Poppins', sans-serif;
    }
    .footer-container a:hover {
        color: #38bdf8;
        text-decoration: underline;
        text-shadow: 0px 0px 8px rgba(56, 189, 248, 0.7);
    }

    /* Streamlit widgets styling */
    .stSelectbox > div > div {
        background-color: #1a222c;
        border: 1px solid #2d333b;
        color: #e6edf3;
        border-radius: 8px;
        padding: 5px 10px;
        font-family: 'Poppins', sans-serif;
    }
    .stSelectbox > label {
        color: #facc15;
        font-size: 19px;
        font-weight: bold;
        font-family: 'Poppins', sans-serif;
        margin-top: 20px; /* Adjusted: Increased top margin for selectbox label */
        display: block; /* Ensure margin applies correctly */
    }
    .stMultiSelect > div > div {
        background-color: #1a222c;
        border: 1px solid #2d333b;
        color: #e6edf3;
        border-radius: 8px;
        padding: 5px 10px;
        font-family: 'Poppins', sans-serif;
    }
    .stMultiSelect > label {
        color: #facc15;
        font-size: 19px;
        font-weight: bold;
        font-family: 'Poppins', sans-serif;
        margin-top: 20px; /* Adjusted: Increased top margin for multiselect label */
        display: block; /* Ensure margin applies correctly */
    }
    .stButton > button {
        background-color: #facc15;
        color: #0f172a;
        font-weight: bold;
        border-radius: 10px;
        padding: 12px 25px;
        border: none;
        transition: all 0.3s ease-in-out;
        font-size: 18px;
        font-family: 'Outfit', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton > button:hover {
        background-color: #fbbf24;
        color: #000;
        box-shadow: 0 0 20px rgba(250, 204, 21, 0.9);
        transform: translateY(-2px);
    }

    /* General text styling */
    h1, h2, h3, h4, h5, h6 {
        color: #facc15;
        font-family: 'Outfit', sans-serif;
        margin-top: 30px; /* Added: More top margin for all headings */
        margin-bottom: 15px; /* Ensure some space below headings */
    }
    p {
        color: #e6edf3;
        font-family: 'Poppins', sans-serif;
    }

    /* Info text for instructions */
    .instruction-text {
        text-align: left;
        font-size: 21px;
        padding-top: 25px; /* Adjusted: More padding at the top */
        color: #94a3b8;
        margin-bottom: 30px; /* Adjusted: More margin at the bottom */
        font-family: 'Poppins', sans-serif;
    }

    .stSuccess {
        background-color: #1a472a;
        color: #6ee7b7;
        border-left: 5px solid #6ee7b7;
        padding: 15px;
        border-radius: 8px;
        font-size: 18px;
        font-weight: 600;
        margin-top: 30px;
        font-family: 'Poppins', sans-serif;
    }
    .stWarning {
        background-color: #4a3212;
        color: #fdba74;
        border-left: 5px solid #fdba74;
        padding: 15px;
        border-radius: 8px;
        font-size: 18px;
        font-weight: 600;
        margin-top: 15px;
        font-family: 'Poppins', sans-serif;
    }

</style>
""", unsafe_allow_html=True)

# --- Placeholder for Model and Data Loading ---
@st.cache_resource(ttl=3600) # Cache model for 1 hour to prevent re-loading on every rerun
def load_ev_model():
    """
    Loads the pre-trained EV forecasting model.
    Replace with your actual model loading logic.
    """
    try:
        model = joblib.load('forecasting_ev_model.pkl')
        return model
    except FileNotFoundError:
        st.error("🚨 Model file 'forecasting_ev_model.pkl' not found. Please ensure it's in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

@st.cache_data(ttl=3600) # Cache data for 1 hour
def load_ev_data():
    """
    Loads and preprocesses the EV data.
    Replace with your actual data loading and preprocessing logic.
    This dummy data simulates the structure your model expects if the file is not found.
    """
    try:
        df = pd.read_csv("preprocessed_ev_data.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        # Ensure 'months_since_start' and 'county_encoded' exist for the data
        if 'months_since_start' not in df.columns:
            # Calculate months_since_start relative to the first date in the *entire* dataset
            min_date = df['Date'].min()
            df['months_since_start'] = ((df['Date'] - min_date).dt.days / 30.4375).astype(int) # Approx days in month
        if 'county_encoded' not in df.columns:
            # Simple encoding for demonstration, ensure it matches your model's training
            df['county_encoded'] = df['County'].astype('category').cat.codes
        return df
    except FileNotFoundError:
        st.warning("⚠️ `preprocessed_ev_data.csv` not found. Generating dummy data for demonstration.")
        # Generate dummy data if the file is not found
        data = {
            'County': ['King', 'King', 'King', 'Pierce', 'Pierce', 'Pierce', 'Snohomish', 'Snohomish', 'King', 'King', 'King', 'King', 'King', 'King', 'Pierce', 'Pierce', 'Pierce', 'Pierce', 'Pierce', 'Pierce', 'Snohomish', 'Snohomish', 'Snohomish', 'Snohomish', 'Snohomish', 'Snohomish'],
            'Date': [
                datetime(2023, 1, 1), datetime(2023, 2, 1), datetime(2023, 3, 1),
                datetime(2023, 1, 1), datetime(2023, 2, 1), datetime(2023, 3, 1),
                datetime(2023, 1, 1), datetime(2023, 2, 1),
                datetime(2023, 4, 1), datetime(2023, 5, 1), datetime(2023, 6, 1), datetime(2023, 7, 1), datetime(2023, 8, 1), datetime(2023, 9, 1),
                datetime(2023, 4, 1), datetime(2023, 5, 1), datetime(2023, 6, 1), datetime(2023, 7, 1), datetime(2023, 8, 1), datetime(2023, 9, 1),
                datetime(2023, 3, 1), datetime(2023, 4, 1), datetime(2023, 5, 1), datetime(2023, 6, 1), datetime(2023, 7, 1), datetime(2023, 8, 1)
            ],
            'Electric Vehicle (EV) Total': [
                1000, 1050, 1100, 500, 520, 540, 700, 730,
                1150, 1200, 1260, 1320, 1380, 1450,
                560, 580, 600, 620, 640, 660,
                760, 790, 820, 850, 880, 910
            ]
        }
        df_dummy = pd.DataFrame(data)
        df_dummy['Date'] = pd.to_datetime(df_dummy['Date'])
        min_date_dummy = df_dummy['Date'].min()
        df_dummy['months_since_start'] = ((df_dummy['Date'] - min_date_dummy).dt.days / 30.4375).astype(int)
        df_dummy['county_encoded'] = df_dummy['County'].astype('category').cat.codes
        return df_dummy.sort_values(by=['County', 'Date']).reset_index(drop=True)
    except Exception as e:
        st.error(f"❌ Error loading or generating data: {e}")
        st.stop()


model = load_ev_model()
df = load_ev_data()


# --- Header ---
st.markdown("""
    <div class="header-container">
        <h1 class="header-title">⚡ EVOLUTION: EV ADOPTION FORECASTER</h1>
        <p class="header-subtitle">🔮 Smart Forecasting for Counties in Washington State</p>
    </div>
""", unsafe_allow_html=True)

# --- Welcome Subtitle ---
st.markdown("""
    <div class="welcome-banner">
        🚗 Welcome to the <span style="color:#ffffff;">Electric Vehicle (EV)</span> Adoption Forecast Tool ⚡<br>
        <span class="sub-text">
            Predict, analyze, and visualize the future of EV growth with intelligent forecasting.
        </span>
    </div>
""", unsafe_allow_html=True)

# --- Image ---
try:
    st.image("ev_car_factory.jpg", use_container_width=True, caption="Driving towards an Electric Future")
except FileNotFoundError:
    st.warning("⚠️ Image 'ev_car_factory.jpg' not found. Please add an image or remove this line.")

# Instruction line
st.markdown("""
    <p class="instruction-text">
        Select a county from the dropdown below to explore its historical EV adoption and our AI-powered 3-year forecast.
        Gain insights into future trends and prepare for the electric revolution!
    </p>
""", unsafe_allow_html=True)

# --- Forecasting Function ---
@st.cache_data(show_spinner="⚡ Generating AI Forecast...", ttl=600) # Cache results for 10 minutes
def get_ev_forecast(county_df_input, county_code_input, _model_input, forecast_horizon=36):
    """
    Performs EV forecasting for a given county dataframe using the provided model.

    Args:
        county_df_input (pd.DataFrame): Historical data for the selected county.
        county_code_input (int): Encoded county ID.
        _model_input: The trained machine learning model (RandomForestRegressor).
                      Prepended with '_' to tell Streamlit not to hash it for caching.
        forecast_horizon (int): Number of months to forecast into the future.

    Returns:
        tuple: (combined_df, historical_df, forecast_only_df)
    """
    # Defensive copy to avoid modifying original cached DataFrame
    county_df = county_df_input.copy()

    # Ensure enough historical data for lags and cumulative calculations
    if len(county_df) < 6:
        st.warning(f"Insufficient historical data ({len(county_df)} months) for accurate forecasting. Need at least 6 months.")
        # Return empty dataframes or stop execution if not enough data
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    historical_ev_monthly = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
    latest_cumulative_ev = county_df['Electric Vehicle (EV) Total'].cumsum().iloc[-1]

    # For slope calculation, use the last 6 cumulative points based on monthly totals
    # (recreating logic from original but ensuring it's robust)
    cumulative_window_for_slope = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
    current_cumulative_values_for_slope = np.cumsum(cumulative_window_for_slope).tolist()

    months_since_start = county_df['months_since_start'].max()
    latest_date = county_df['Date'].max()

    future_rows = []
    current_cumulative_val = latest_cumulative_ev # Start cumulative forecast from last historical cumulative

    for i in range(1, forecast_horizon + 1):
        forecast_date = latest_date + pd.DateOffset(months=i)
        months_since_start += 1

        # Prepare features for prediction
        lag1 = historical_ev_monthly[-1] if len(historical_ev_monthly) >= 1 else 0
        lag2 = historical_ev_monthly[-2] if len(historical_ev_monthly) >= 2 else 0
        lag3 = historical_ev_monthly[-3] if len(historical_ev_monthly) >= 3 else 0

        roll_mean = np.mean([lag1, lag2, lag3]) if (lag1 and lag2 and lag3) else lag1 # Fallback if not enough lags
        pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
        pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0

        ev_growth_slope = 0
        if len(current_cumulative_values_for_slope) >= 2: # Need at least 2 points for a slope
            # Generate x-values for polyfit (e.g., [0, 1, 2, 3, 4, 5] for 6 points)
            x_values = range(len(current_cumulative_values_for_slope))
            ev_growth_slope = np.polyfit(x_values, current_cumulative_values_for_slope, 1)[0]


        new_row_features = {
            'months_since_start': months_since_start,
            'county_encoded': county_code_input,
            'ev_total_lag1': lag1,
            'ev_total_lag2': lag2,
            'ev_total_lag3': lag3,
            'ev_total_roll_mean_3': roll_mean,
            'ev_total_pct_change_1': pct_change_1,
            'ev_total_pct_change_3': pct_change_3,
            'ev_growth_slope': ev_growth_slope
        }

        # Predict the monthly EV count using the unhashed model
        pred_monthly = _model_input.predict(pd.DataFrame([new_row_features]))[0]
        pred_monthly = max(0, round(pred_monthly)) # Ensure non-negative forecast

        current_cumulative_val += pred_monthly # Add monthly prediction to cumulative total
        future_rows.append({"Date": forecast_date, "Predicted EV Total": pred_monthly, "Cumulative EV": current_cumulative_val})

        # Update historical_ev_monthly for next iteration's lags
        historical_ev_monthly.append(pred_monthly)
        if len(historical_ev_monthly) > 6:
            historical_ev_monthly.pop(0)

        # Update cumulative_values_for_slope for next iteration's slope calculation
        current_cumulative_values_for_slope.append(pred_monthly) # Append the monthly prediction itself
        if len(current_cumulative_values_for_slope) > 6:
            current_cumulative_values_for_slope.pop(0)


    # Prepare historical data for plotting
    historical_cum = county_df[['Date', 'Electric Vehicle (EV) Total']].copy()
    historical_cum['Source'] = 'Historical'
    historical_cum['Cumulative EV'] = historical_cum['Electric Vehicle (EV) Total'].cumsum()

    # Prepare forecast data for plotting
    forecast_df = pd.DataFrame(future_rows)
    forecast_df['Source'] = 'Forecast'
    # The 'Cumulative EV' in forecast_df is already correctly calculated in the loop

    # Combine historical and forecasted data for a single plot
    combined_df = pd.concat([
        historical_cum[['Date', 'Cumulative EV', 'Source']],
        forecast_df[['Date', 'Cumulative EV', 'Source']]
    ], ignore_index=True)

    return combined_df, historical_cum, forecast_df


# --- Main Application Layout ---
county_list = sorted(df['County'].dropna().unique().tolist())
selected_county = st.selectbox("Select a County", county_list, key="single_county_select")

if selected_county:
    county_df_filtered = df[df['County'] == selected_county].sort_values("Date")

    if county_df_filtered.empty or len(county_df_filtered) < 6: # Check for insufficient data
        st.warning(f"⚠️ Not enough historical data available for '{selected_county}' to generate a meaningful forecast. Need at least 6 months of data. Please select another county.")
        st.stop()

    county_code = county_df_filtered['county_encoded'].iloc[0]

    # --- Generate Forecast ---
    # Pass the 'model' object (which is loaded via @st.cache_resource)
    # The get_ev_forecast function will receive it as '_model_input' and ignore it for its own caching.
    combined_forecast_df, historical_df, forecast_only_df = get_ev_forecast(
        county_df_filtered, county_code, model
    )

    st.markdown("---")
    st.header(f"📊 Cumulative EV Adoption Trend for {selected_county} County")

    # --- Plot Cumulative Graph with Plotly ---
    fig_single = go.Figure()

    # Add Historical Trace
    fig_single.add_trace(go.Scatter(
        x=historical_df['Date'],
        y=historical_df['Cumulative EV'],
        mode='lines+markers',
        name='Historical Data',
        line=dict(color='#38bdf8', width=4), # Vibrant blue, thicker line
        marker=dict(size=7, color='#38bdf8', symbol='circle', line=dict(width=1.5, color='#ffffff')), # Larger, clear markers
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Historical EVs:</b> %{y:,.0f}<extra></extra>'
    ))

    # Add Forecast Trace
    fig_single.add_trace(go.Scatter(
        x=forecast_only_df['Date'],
        y=forecast_only_df['Cumulative EV'],
        mode='lines+markers',
        name='Forecasted Trend',
        line=dict(color='#facc15', width=4, dash='dashdot'), # Golden yellow, dashed line, thicker
        marker=dict(size=7, color='#facc15', symbol='diamond', line=dict(width=1.5, color='#ffffff')), # Larger, distinct markers
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Forecasted EVs:</b> %{y:,.0f}<extra></extra>'
    ))

    fig_single.update_layout(
        title={
            'text': f"Cumulative EV Trend for <span style='color:#38bdf8;'>{selected_county}</span> (Historical + 3 Years Forecast)",
            'y':0.98, # Adjusted: Pushed title higher
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=26, color='#facc15', family='Outfit, sans-serif') # Larger, prominent title
        },
        xaxis_title="Date",
        yaxis_title="Cumulative EV Count",
        hovermode="x unified",
        paper_bgcolor="#1c1c1c", # Dark background for the plot area
        plot_bgcolor="#1c1c1c", # Dark background for the plot itself
        font=dict(color="#e6edf3", family='Poppins, sans-serif'), # Combined font families into a single string
        xaxis=dict(
            gridcolor='#2d333b',
            tickfont=dict(size=14), # Larger tick labels
            title_font=dict(size=18)
        ),
        yaxis=dict(
            gridcolor='#2d333b',
            tickfont=dict(size=14), # Larger tick labels
            title_font=dict(size=18),
            rangemode='tozero' # Ensure y-axis starts from 0
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(28,28,28,0.8)", # Semi-transparent background for legend
            bordercolor="#facc15",
            borderwidth=1,
            font=dict(color="#e6edf3", size=15) # Larger legend font
        ),
        margin=dict(l=40, r=40, t=100, b=40) # Adjusted: Increased top margin for title
    )
    st.plotly_chart(fig_single, use_container_width=True)

    # --- Growth Insight Block ---
    historical_total = historical_df['Cumulative EV'].iloc[-1] if not historical_df.empty else 0
    forecasted_total = forecast_only_df['Cumulative EV'].iloc[-1] if not forecast_only_df.empty else historical_total # If no forecast, use historical

    if historical_total > 0:
        forecast_growth_pct = ((forecasted_total - historical_total) / historical_total) * 100
        trend_icon = "📈" if forecast_growth_pct > 0 else "📉"
        trend_text = "Increase" if forecast_growth_pct > 0 else "Decrease"
        st.markdown(f"""
            <div class="growth-insight-block">
                <h2 style="color:#facc15; margin-bottom:10px;">
                    {trend_icon} Projected <span style="color:#38bdf8;">{trend_text}</span> in EV Adoption for <span style="color:#38bdf8;'>{selected_county}</span>
                </h2>
                <p class="metric">
                    <b>{forecast_growth_pct:.2f}%</b> projected growth over the next <b>3 years</b>
                </p>
                <p class="subtitle">
                    Our AI-based forecast provides critical insights for smarter planning and infrastructure development.
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("ℹ️ Historical EV total is zero or insufficient data, so percentage forecast change can't be computed for this county.")


# --- Compare Multiple Counties ---
st.markdown("---")
st.header("🤝 Compare EV Adoption Trends for up to 3 Counties")

multi_counties = st.multiselect(
    "Select up to 3 counties to compare their EV adoption trajectories:",
    county_list,
    max_selections=3,
    key="multi_county_select"
)

if multi_counties:
    comparison_data_frames = []
    for cty in multi_counties:
        cty_df_filtered = df[df['County'] == cty].sort_values("Date")
        if not cty_df_filtered.empty and len(cty_df_filtered) >= 6: # Ensure enough data for comparison forecast
            cty_code = cty_df_filtered['county_encoded'].iloc[0]
            # Pass 'model' to get_ev_forecast which expects '_model_input'
            combined_cty_forecast, _, _ = get_ev_forecast(cty_df_filtered, cty_code, model)
            combined_cty_forecast['County'] = cty # Add county column for plotting
            comparison_data_frames.append(combined_cty_forecast)
        else:
            st.warning(f"⚠️ Not enough historical data available for '{cty}' to include in comparison. Need at least 6 months of data.")

    if comparison_data_frames:
        comp_df = pd.concat(comparison_data_frames, ignore_index=True)

        # Plotly for comparison
        st.subheader("📈 Comparison of Cumulative EV Adoption Trends (Historical + 3-Year Forecast)")
        fig_multi = go.Figure()

        # Using Plotly Express color schemes for more variety
        color_palette = px.colors.qualitative.Plotly + px.colors.qualitative.T10

        for i, cty in enumerate(multi_counties):
            cty_data = comp_df[comp_df['County'] == cty]
            plot_color = color_palette[i % len(color_palette)]

            # Add Historical Trace for this county
            hist_data = cty_data[cty_data['Source'] == 'Historical']
            fig_multi.add_trace(go.Scatter(
                x=hist_data['Date'],
                y=hist_data['Cumulative EV'],
                mode='lines+markers',
                name=f'{cty} (Historical)',
                line=dict(color=plot_color, width=3),
                marker=dict(size=6, color=plot_color, symbol='circle', line=dict(width=1, color='#ffffff')),
                legendgroup=cty, # Group legend entries for the same county
                hovertemplate=f'<b>County:</b> {cty}<br><b>Date:</b> %{{x|%Y-%m-%d}}<br><b>Historical EVs:</b> %{{y:,.0f}}<extra></extra>'
            ))

            # Add Forecast Trace for this county
            forecast_data = cty_data[cty_data['Source'] == 'Forecast']
            fig_multi.add_trace(go.Scatter(
                x=forecast_data['Date'],
                y=forecast_data['Cumulative EV'],
                mode='lines+markers',
                name=f'{cty} (Forecast)',
                line=dict(color=plot_color, width=3, dash='dash'), # Dashed for forecast
                marker=dict(size=6, color=plot_color, symbol='diamond', line=dict(width=1, color='#ffffff')),
                legendgroup=cty, # Group legend entries
                showlegend=False, # Don't duplicate legend entry for forecast
                hovertemplate=f'<b>County:</b> {cty}<br><b>Date:</b> %{{x|%Y-%m-%d}}<br><b>Forecasted EVs:</b> %{{y:,.0f}}<extra></extra>'
            ))


        fig_multi.update_layout(
            title={
                'text': "EV Adoption Trends: Historical + 3-Year Forecast Across Selected Counties",
                'y':0.98, # Adjusted: Pushed title higher
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=26, color='#facc15', family='Outfit, sans-serif')
            },
            xaxis_title="Date",
            yaxis_title="Cumulative EV Count",
            hovermode="x unified",
            paper_bgcolor="#1c1c1c",
            plot_bgcolor="#1c1c1c",
            font=dict(color="#e6edf3", family='Poppins, sans-serif'), # Combined font families into a single string
            xaxis=dict(
                gridcolor='#2d333b',
                tickfont=dict(size=14),
                title_font=dict(size=18)
            ),
            yaxis=dict(
                gridcolor='#2d333b',
                tickfont=dict(size=14),
                title_font=dict(size=18),
                rangemode='tozero'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(28,28,28,0.8)",
                bordercolor="#facc15",
                borderwidth=1,
                font=dict(color="#e6edf3", size=15)
            ),
            margin=dict(l=40, r=40, t=100, b=40) # Adjusted: Increased top margin for title
        )
        st.plotly_chart(fig_multi, use_container_width=True)

        # --- Forecasted Growth Summary Block for Comparison ---
        growth_summaries = []
        for cty in multi_counties:
            cty_df_for_growth = comp_df[comp_df['County'] == cty]
            # Ensure we have data for the selected county in the comparison_data_frames list
            if not cty_df_for_growth.empty:
                # Find the last historical point and the last forecasted point within this specific county's data
                last_historical_date_for_county = df[df['County'] == cty]['Date'].max()
                historical_data_point = cty_df_for_growth[cty_df_for_growth['Date'] == last_historical_date_for_county]

                historical_total = historical_data_point['Cumulative EV'].iloc[-1] if not historical_data_point.empty else 0
                forecasted_total = cty_df_for_growth['Cumulative EV'].iloc[-1]

                if historical_total > 0:
                    growth_pct = ((forecasted_total - historical_total) / historical_total) * 100
                    icon = "📈" if growth_pct > 0 else "📉"
                    color = "#22c55e" if growth_pct > 0 else "#ef4444"
                    growth_summaries.append(f"<span style='color:{color}; font-weight:bold;'>{icon} {cty}: {growth_pct:.2f}%</span>")
                else:
                    growth_summaries.append(f"<span style='color:#94a3b8;'>{cty}: N/A (Insufficient historical base)</span>")

        if growth_summaries:
            growth_sentence = " | ".join(growth_summaries)
            st.markdown(f"""
                <div class="comparison-block">
                    <h2>
                        🚀 Forecasted EV Adoption Growth (Next 3 Years)
                    </h2>
                    <p>
                        {growth_sentence}
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ No comparable data for selected counties to show growth summaries.")


# --- Footer ---
st.markdown("""
    <hr style="border: 1px solid #2d333b; margin-top: 50px; margin-bottom: 25px;">
    <div class="footer-container">
        <p>
            👨‍💻 Developed by: Anurag Pratap Singh
        </p>
        <a href="https://github.com/Anurag7321-singh/EV-Charging-Prediction-Model"
            target="_blank">
            🌐 Visit GitHub Repository
        </a>
    </div>
""", unsafe_allow_html=True)

st.success("✅ EV Adoption Forecast analysis complete!")
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 16px; margin-top: 15px; font-family: \"Poppins\", sans-serif;'>Prepared for the <b>AICTE Internship Cycle 2 by S4F</b></p>", unsafe_allow_html=True)