from pydantic import BaseModel, Field, PrivateAttr
from typing import Literal, Optional
from datetime import datetime
import pandas as pd

import torch


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
    humidity:          float = Field(..., ge=0)
    precip:            float = Field(..., ge=0)
    precip_prob:       int   = Field(..., ge=0, le=100)
    wind_gust:         float = Field(..., ge=0)
    wind_speed:        float = Field(..., ge=0)
    cloud_cover:       float = Field(..., ge=0)
    solar_radiation:   float = Field(..., ge=0)
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

    _invalid_reason: Optional[str] = PrivateAttr(None)

    def to_df(self) -> pd.DataFrame:
        # Step 1: Start from base dict
        base = self.model_dump()

        # Step 2: Construct date
        date_obj = datetime(self.year, self.month, self.day)

        # Step 3: Add derived time features
        base["day_of_year"] = date_obj.timetuple().tm_yday
        base["is_Monday"] = int(date_obj.weekday() == 0)
        base["is_Tuesday"] = int(date_obj.weekday() == 1)
        base["is_Wednesday"] = int(date_obj.weekday() == 2)
        base["is_Thursday"] = int(date_obj.weekday() == 3)
        base["is_Friday"] = int(date_obj.weekday() == 4)
        base["is_Saturday"] = int(date_obj.weekday() == 5)
        base["is_Sunday"] = int(date_obj.weekday() == 6)

        # Step 4: Handle one-hot holiday columns
        holidays = [
            "Ascension Day", "Christmas", "Day of German Unity",
            "Easter Monday", "Good Friday", "King's Day",
            "May Day", "New Year's Day", "Second Christmas Day",
            "Whit Monday"
        ]
        for h in holidays:
            base[h] = int(self.holiday == h)

        # Step 5: Remove unused original holiday & date info
        base.pop("holiday", None)
        base.pop("day", None)  # no need anymore

        # Step 6: Rename keys to match training feature names
        rename_map = {
            "temp_max": "tempmax",
            "temp_min": "tempmin",
            "feels_like_max": "feelslikemax",
            "feels_like_min": "feelslikemin",
            "feels_like": "feelslike",
            "cloud_cover": "cloudcover",
            "solar_radiation": "solarradiation",
            "uv_index": "uvindex",
            "precip_prob": "precipprob",
            "wind_gust": "windgust",
            "wind_speed": "windspeed",
            "is_school_holiday": "IsHoliday",
        }

        for old_name, new_name in rename_map.items():
            base[new_name] = base.pop(old_name)

        df = pd.DataFrame([base])
        # Order the columns, so the order always matches
        df = df.reindex(sorted(df.columns), axis=1)
        return df
    
    def to_tensor(self) -> torch.Tensor:
        """
        Converts the model input to a PyTorch tensor.
        
        Returns:
            torch.Tensor: A tensor representation of the model input.
        """
        df = self.to_df()
        return torch.tensor(df.to_numpy(), dtype=torch.float32)

    
    def is_valid(self) -> bool:
        """Performs logical validation of the input values."""
        try:
            # Valid date
            datetime(self.year, self.month, self.day)

            # Logical temperature consistency
            if not self.temp_min <= self.temp <= self.temp_max:
                self._invalid_reason = "temp not between min and max"
                return False
            if not self.feels_like_min <= self.feels_like <= self.feels_like_max:
                self._invalid_reason = "feels_like not between min and max"
                return False

            # Wind consistency
            if self.wind_gust < self.wind_speed:
                self._invalid_reason = "self.wind_gust < self.wind_speed"
                return False

            return True
        except Exception:
            self._invalid_reason = "invalid date"
            return False

    @property
    def invalid_reason(self):
        return self._invalid_reason
