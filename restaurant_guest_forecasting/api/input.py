from pydantic import BaseModel, Field
from typing import Literal, Optional


class ModelInput(BaseModel):
    """
    DTO holding model input.
    """
    day:               int   = Field(..., ge=1, le=31)
    month:             int   = Field(..., ge=1, le=12)
    year:              int   = Field(..., ge=2019)
    # from date we can infere day of week
    temp_max:          float
    temp_min:          float
    temp:              float
    feels_like_max:    float
    feels_like_min:    float
    feels_like:        float
    humidity:          float
    precip:            float
    precip_prob:       int   = Field(..., ge=0, le=100)
    wind_gust:         float
    wind_speed:        float
    cloud_cover:       float
    solar_radiation:   float
    uv_index:          int   = Field(..., ge=0)
    rain:              int   = Field(..., ge=0, le=1)         
    snow:              int   = Field(..., ge=0, le=1)
    is_school_holiday: int   = Field(..., ge=0, le=1) # IsHoliday column

    # Optional holiday string with specific allowed values
    holiday: Optional[
        Literal[
            "Ascension Day", "Christmas", "Day of German Unity",
            "Easter Monday", "Good Friday", "King's Day",
            "May Day", "New Year's Day", "Second Christmas Day",
            "Whit Monday"
        ]
    ]