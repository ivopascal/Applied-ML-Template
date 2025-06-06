import streamlit as st
import pandas as pd
from datetime import date

def predict(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Call the predictor here and return a new df with features for date and the predicted value.
    """
    predicted_df = data_df.copy()
    predicted_df = predicted_df[['day', 'month', 'year']]
    predicted_df['target'] = (data_df['temp'] * 0.5 + data_df['humidity'] * 0.1 - data_df['precip'] * 2).round(0)
    return predicted_df

st.set_page_config(layout="wide")
st.title("Restaurant Guests Forecasting App")

st.header("Input Data for Prediction")

# Column definitions for input
col_defs = [
    {"name": "day", "type": "number", "min_value": 1, "max_value": 31, "step": 1, "format": "%d", "default": date.today().day},
    {"name": "month", "type": "number", "min_value": 1, "max_value": 12, "step": 1, "format": "%d", "default": date.today().month},
    {"name": "year", "type": "number", "min_value": 1900, "max_value": 2100, "step": 1, "format": "%d", "default": date.today().year},
    {"name": "temp_max", "type": "number", "min_value": -50.0, "max_value": 50.0, "step": 0.1, "format": "%.1f", "default": 15.0},
    {"name": "temp_min", "type": "number", "min_value": -50.0, "max_value": 50.0, "step": 0.1, "format": "%.1f", "default": 5.0},
    {"name": "temp", "type": "number", "min_value": -50.0, "max_value": 50.0, "step": 0.1, "format": "%.1f", "default": 10.0},
    {"name": "feels_like_max", "type": "number", "min_value": -50.0, "max_value": 50.0, "step": 0.1, "format": "%.1f", "default": 14.0},
    {"name": "feels_like_min", "type": "number", "min_value": -50.0, "max_value": 50.0, "step": 0.1, "format": "%.1f", "default": 4.0},
    {"name": "feels_like", "type": "number", "min_value": -50.0, "max_value": 50.0, "step": 0.1, "format": "%.1f", "default": 9.0},
    {"name": "humidity", "type": "number", "min_value": 0.0, "max_value": 100.0, "step": 0.1, "format": "%.1f", "default": 70.0},
    {"name": "precip", "type": "number", "min_value": 0.0, "max_value": 500.0, "step": 0.1, "format": "%.1f", "default": 0.0},
    {"name": "precip_prob", "type": "number", "min_value": 0.0, "max_value": 100.0, "step": 0.1, "format": "%.1f", "default": 10.0},
    {"name": "wind_gust", "type": "number", "min_value": 0.0, "max_value": 300.0, "step": 0.1, "format": "%.1f", "default": 20.0},
    {"name": "wind_speed", "type": "number", "min_value": 0.0, "max_value": 300.0, "step": 0.1, "format": "%.1f", "default": 10.0},
    {"name": "cloud_cover", "type": "number", "min_value": 0.0, "max_value": 100.0, "step": 0.1, "format": "%.1f", "default": 50.0},
    {"name": "solar_radiation", "type": "number", "min_value": 0.0, "max_value": 2000.0, "step": 0.1, "format": "%.1f", "default": 150.0},
    {"name": "uv_index", "type": "number", "min_value": 0, "max_value": 11, "step": 1, "format": "%d", "default": 3},
    {"name": "rain", "type": "select", "options": [0, 1], "default": 0},
    {"name": "snow", "type": "select", "options": [0, 1], "default": 0},
    {"name": "is_school_holiday", "type": "select", "options": [0, 1], "default": 0},
    {"name": "holiday", "type": "select", "options": ["None", "Ascension Day", "Christmas", "Day of German Unity", "Easter Monday", "Good Friday", "King's Day", "May Day", "New Year's Day", "Second Christmas Day", "Whit Monday"], "default": "None"}
]

# Initialize input_data in session state if it doesn't exist
if 'input_data' not in st.session_state:
    st.session_state.input_data = {}
    # Populate with default values on first run
    for col_def in col_defs:
        st.session_state.input_data[col_def["name"]] = col_def["default"]

# Arrange inputs on multiple rows using columns
# Group related inputs together
# Row 1: Date
col1, col2, col3 = st.columns(3)
with col1:
    st.session_state.input_data["day"] = st.number_input(
        label="Day",
        min_value=col_defs[0]["min_value"],
        max_value=col_defs[0]["max_value"],
        value=int(st.session_state.input_data["day"]), # Ensure int type for date parts
        step=col_defs[0]["step"],
        format=col_defs[0]["format"],
        key="day_input"
    )
with col2:
    st.session_state.input_data["month"] = st.number_input(
        label="Month",
        min_value=col_defs[1]["min_value"],
        max_value=col_defs[1]["max_value"],
        value=int(st.session_state.input_data["month"]),
        step=col_defs[1]["step"],
        format=col_defs[1]["format"],
        key="month_input"
    )
with col3:
    st.session_state.input_data["year"] = st.number_input(
        label="Year",
        min_value=col_defs[2]["min_value"],
        max_value=col_defs[2]["max_value"],
        value=int(st.session_state.input_data["year"]),
        step=col_defs[2]["step"],
        format=col_defs[2]["format"],
        key="year_input"
    )

st.markdown("---") # Separator

# Row 2: Temperatures (Max, Min, Avg, Feels Like)
col4, col5, col6 = st.columns(3)
with col4:
    st.session_state.input_data["temp_max"] = st.number_input(
        label="Temp Max (°C)",
        min_value=col_defs[3]["min_value"],
        max_value=col_defs[3]["max_value"],
        value=float(st.session_state.input_data["temp_max"]),
        step=col_defs[3]["step"],
        format=col_defs[3]["format"],
        key="temp_max_input"
    )
with col5:
    st.session_state.input_data["temp_min"] = st.number_input(
        label="Temp Min (°C)",
        min_value=col_defs[4]["min_value"],
        max_value=col_defs[4]["max_value"],
        value=float(st.session_state.input_data["temp_min"]),
        step=col_defs[4]["step"],
        format=col_defs[4]["format"],
        key="temp_min_input"
    )
with col6:
    st.session_state.input_data["temp"] = st.number_input(
        label="Temp Avg (°C)",
        min_value=col_defs[5]["min_value"],
        max_value=col_defs[5]["max_value"],
        value=float(st.session_state.input_data["temp"]),
        step=col_defs[5]["step"],
        format=col_defs[5]["format"],
        key="temp_avg_input"
    )

col7, col8, col9 = st.columns(3)
with col7:
    st.session_state.input_data["feels_like_max"] = st.number_input(
        label="Feels Like Max (°C)",
        min_value=col_defs[6]["min_value"],
        max_value=col_defs[6]["max_value"],
        value=float(st.session_state.input_data["feels_like_max"]),
        step=col_defs[6]["step"],
        format=col_defs[6]["format"],
        key="feels_like_max_input"
    )
with col8:
    st.session_state.input_data["feels_like_min"] = st.number_input(
        label="Feels Like Min (°C)",
        min_value=col_defs[7]["min_value"],
        max_value=col_defs[7]["max_value"],
        value=float(st.session_state.input_data["feels_like_min"]),
        step=col_defs[7]["step"],
        format=col_defs[7]["format"],
        key="feels_like_min_input"
    )
with col9:
    st.session_state.input_data["feels_like"] = st.number_input(
        label="Feels Like Avg (°C)",
        min_value=col_defs[8]["min_value"],
        max_value=col_defs[8]["max_value"],
        value=float(st.session_state.input_data["feels_like"]),
        step=col_defs[8]["step"],
        format=col_defs[8]["format"],
        key="feels_like_avg_input"
    )

st.markdown("---") # Separator

# Row 3: Humidity, Precipitation, Wind
col10, col11, col12 = st.columns(3)
with col10:
    st.session_state.input_data["humidity"] = st.number_input(
        label="Humidity (%)",
        min_value=col_defs[9]["min_value"],
        max_value=col_defs[9]["max_value"],
        value=float(st.session_state.input_data["humidity"]),
        step=col_defs[9]["step"],
        format=col_defs[9]["format"],
        key="humidity_input"
    )
with col11:
    st.session_state.input_data["precip"] = st.number_input(
        label="Precipitation (mm)",
        min_value=col_defs[10]["min_value"],
        max_value=col_defs[10]["max_value"],
        value=float(st.session_state.input_data["precip"]),
        step=col_defs[10]["step"],
        format=col_defs[10]["format"],
        key="precip_input"
    )
with col12:
    st.session_state.input_data["precip_prob"] = st.number_input(
        label="Precip. Prob (%)",
        min_value=col_defs[11]["min_value"],
        max_value=col_defs[11]["max_value"],
        value=float(st.session_state.input_data["precip_prob"]),
        step=col_defs[11]["step"],
        format=col_defs[11]["format"],
        key="precip_prob_input"
    )

col13, col14 = st.columns(2)
with col13:
    st.session_state.input_data["wind_gust"] = st.number_input(
        label="Wind Gust (km/h)",
        min_value=col_defs[12]["min_value"],
        max_value=col_defs[12]["max_value"],
        value=float(st.session_state.input_data["wind_gust"]),
        step=col_defs[12]["step"],
        format=col_defs[12]["format"],
        key="wind_gust_input"
    )
with col14:
    st.session_state.input_data["wind_speed"] = st.number_input(
        label="Wind Speed (km/h)",
        min_value=col_defs[13]["min_value"],
        max_value=col_defs[13]["max_value"],
        value=float(st.session_state.input_data["wind_speed"]),
        step=col_defs[13]["step"],
        format=col_defs[13]["format"],
        key="wind_speed_input"
    )

st.markdown("---") # Separator

# Row 4: Cloud Cover, Solar Radiation, UV Index
col15, col16, col17 = st.columns(3)
with col15:
    st.session_state.input_data["cloud_cover"] = st.number_input(
        label="Cloud Cover (%)",
        min_value=col_defs[14]["min_value"],
        max_value=col_defs[14]["max_value"],
        value=float(st.session_state.input_data["cloud_cover"]),
        step=col_defs[14]["step"],
        format=col_defs[14]["format"],
        key="cloud_cover_input"
    )
with col16:
    st.session_state.input_data["solar_radiation"] = st.number_input(
        label="Solar Radiation (W/m^2)",
        min_value=col_defs[15]["min_value"],
        max_value=col_defs[15]["max_value"],
        value=float(st.session_state.input_data["solar_radiation"]),
        step=col_defs[15]["step"],
        format=col_defs[15]["format"],
        key="solar_radiation_input"
    )
with col17:
    st.session_state.input_data["uv_index"] = st.number_input(
        label="UV Index",
        min_value=col_defs[16]["min_value"],
        max_value=col_defs[16]["max_value"],
        value=int(st.session_state.input_data["uv_index"]),
        step=col_defs[16]["step"],
        format=col_defs[16]["format"],
        key="uv_index_input"
    )

st.markdown("---") # Separator

# Row 5: Rain, Snow, School Holiday, Holiday Name
col18, col19, col20 = st.columns(3)
with col18:
    st.session_state.input_data["rain"] = st.selectbox(
        label="Rain (0=No, 1=Yes)",
        options=col_defs[17]["options"],
        index=col_defs[17]["options"].index(st.session_state.input_data["rain"]),
        key="rain_input"
    )
with col19:
    st.session_state.input_data["snow"] = st.selectbox(
        label="Snow (0=No, 1=Yes)",
        options=col_defs[18]["options"],
        index=col_defs[18]["options"].index(st.session_state.input_data["snow"]),
        key="snow_input"
    )
with col20:
    st.session_state.input_data["is_school_holiday"] = st.selectbox(
        label="School Holiday (0=No, 1=Yes)",
        options=col_defs[19]["options"],
        index=col_defs[19]["options"].index(st.session_state.input_data["is_school_holiday"]),
        key="is_school_holiday_input"
    )

st.session_state.input_data["holiday"] = st.selectbox(
    label="Holiday Name",
    options=col_defs[20]["options"],
    index=col_defs[20]["options"].index(st.session_state.input_data["holiday"]),
    key="holiday_input"
)

st.markdown("---") # Separator

# Prediction button
if st.button("Predict guests"):
    input_df = pd.DataFrame([st.session_state.input_data])

    # Ensure correct data types (important for models)
    input_df["day"] = input_df["day"].astype(int)
    input_df["month"] = input_df["month"].astype(int)
    input_df["year"] = input_df["year"].astype(int)
    input_df["temp_max"] = input_df["temp_max"].astype(float).round(1)
    input_df["temp_min"] = input_df["temp_min"].astype(float).round(1)
    input_df["temp"] = input_df["temp"].astype(float).round(1)
    input_df["feels_like_max"] = input_df["feels_like_max"].astype(float).round(1)
    input_df["feels_like_min"] = input_df["feels_like_min"].astype(float).round(1)
    input_df["feels_like"] = input_df["feels_like"].astype(float).round(1)
    input_df["humidity"] = input_df["humidity"].astype(float).round(1)
    input_df["precip"] = input_df["precip"].astype(float).round(1)
    input_df["precip_prob"] = input_df["precip_prob"].astype(float).round(1)
    input_df["wind_gust"] = input_df["wind_gust"].astype(float).round(1)
    input_df["wind_speed"] = input_df["wind_speed"].astype(float).round(1)
    input_df["cloud_cover"] = input_df["cloud_cover"].astype(float).round(1)
    input_df["solar_radiation"] = input_df["solar_radiation"].astype(float).round(1)
    input_df["uv_index"] = input_df["uv_index"].astype(int)
    input_df["rain"] = input_df["rain"].astype(int)
    input_df["snow"] = input_df["snow"].astype(int)
    input_df["is_school_holiday"] = input_df["is_school_holiday"].astype(int)
    # 'holiday' column is already string/object, no specific cast needed for this example

    # st.subheader("Input Data for Prediction:")
    # st.dataframe(input_df)

    try:
        predictions_df = predict(input_df.copy()) # Pass a copy to avoid modifying original input_df
        st.subheader("Prediction Results:")
        
        predicted_row = predictions_df.iloc[0]
        try:
            predicted_date = date(int(predicted_row['year']), int(predicted_row['month']), int(predicted_row['day']))
            st.success(f"**Date:** {predicted_date.strftime('%Y-%m-%d')}, **Predicted Target:** {predicted_row['target']:.1f}")
        except ValueError as e:
            st.error(f"Error creating date: {e}. Data: Day={predicted_row['day']}, Month={predicted_row['month']}, Year={predicted_row['year']}")
            st.error(f"Predicted Target: {predicted_row['target']:.1f}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
