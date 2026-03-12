# ----------------------------- DATA SCHEMA - MODULE ĐỊNH NGHĨA SCHEMA DỮ LIỆU -------------------------------------------------
"""
Schema.py - Module định nghĩa schema cho dữ liệu thời tiết

Mục đích:
    - Định nghĩa cấu trúc dữ liệu cho các bản ghi thời tiết
    - Validation dữ liệu đầu vào
    - Chuyển đổi kiểu dữ liệu
    - Hỗ trợ serialization/deserialization

Cách sử dụng:
    from Weather_Forcast_App.Machine_learning_model.data.Schema import WeatherDataSchema

    # Validate dữ liệu
    schema = WeatherDataSchema(**data_dict)
    validated_data = schema.dict()

    # Hoặc từ DataFrame
    df = pd.DataFrame(...)
    records = [WeatherDataSchema(**row.to_dict()) for _, row in df.iterrows()]
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

import pandas as pd
from pydantic import BaseModel, Field, validator
from enum import Enum

logger = logging.getLogger(__name__)


# ============================= ENUMS =============================

class DataSource(Enum):
    """Nguồn dữ liệu."""
    OPENMETEO = "openmeteo"
    VRAIN = "vrain"
    OTHER = "other"


class DataQuality(Enum):
    """Chất lượng dữ liệu."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ============================= BASE SCHEMAS =============================

class LocationSchema(BaseModel):
    """Schema cho thông tin vị trí."""
    station_id: str = Field(..., description="Station ID")
    station_name: str = Field(..., description="Station name")
    province: str = Field(..., description="Province")
    district: str = Field(..., description="District")
    latitude: float = Field(..., ge=-90, le=90, description="Latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude")


class WeatherMetricsSchema(BaseModel):
    """Schema cho các chỉ số thời tiết."""
    # Temperature (°C)
    temperature_current: Optional[float] = Field(None, description="Current temperature")
    temperature_max: Optional[float] = Field(None, description="Max temperature")
    temperature_min: Optional[float] = Field(None, description="Min temperature")
    temperature_avg: Optional[float] = Field(None, description="Average temperature")

    # Humidity (%)
    humidity_current: Optional[float] = Field(None, ge=0, le=100, description="Current humidity")
    humidity_max: Optional[float] = Field(None, ge=0, le=100, description="Max humidity")
    humidity_min: Optional[float] = Field(None, ge=0, le=100, description="Min humidity")
    humidity_avg: Optional[float] = Field(None, ge=0, le=100, description="Average humidity")

    # Pressure (hPa)
    pressure_current: Optional[float] = Field(None, description="Current pressure")
    pressure_max: Optional[float] = Field(None, description="Max pressure")
    pressure_min: Optional[float] = Field(None, description="Min pressure")
    pressure_avg: Optional[float] = Field(None, description="Average pressure")

    # Wind speed (m/s)
    wind_speed_current: Optional[float] = Field(None, ge=0, description="Current wind speed")
    wind_speed_max: Optional[float] = Field(None, ge=0, description="Max wind speed")
    wind_speed_min: Optional[float] = Field(None, ge=0, description="Min wind speed")
    wind_speed_avg: Optional[float] = Field(None, ge=0, description="Average wind speed")

    # Wind direction (degrees)
    wind_direction_current: Optional[float] = Field(None, ge=0, le=360, description="Current wind direction")
    wind_direction_avg: Optional[float] = Field(None, ge=0, le=360, description="Average wind direction")

    # Rain (mm)
    rain_current: Optional[float] = Field(None, ge=0, description="Current rain")
    rain_max: Optional[float] = Field(None, ge=0, description="Max rain")
    rain_min: Optional[float] = Field(None, ge=0, description="Min rain")
    rain_avg: Optional[float] = Field(None, ge=0, description="Average rain")
    rain_total: Optional[float] = Field(None, ge=0, description="Total rain")

    # Cloud cover (%)
    cloud_cover_current: Optional[float] = Field(None, ge=0, le=100, description="Current cloud cover")
    cloud_cover_max: Optional[float] = Field(None, ge=0, le=100, description="Max cloud cover")
    cloud_cover_min: Optional[float] = Field(None, ge=0, le=100, description="Min cloud cover")
    cloud_cover_avg: Optional[float] = Field(None, ge=0, le=100, description="Average cloud cover")

    # Visibility (km)
    visibility_current: Optional[float] = Field(None, ge=0, description="Current visibility")
    visibility_max: Optional[float] = Field(None, ge=0, description="Max visibility")
    visibility_min: Optional[float] = Field(None, ge=0, description="Min visibility")
    visibility_avg: Optional[float] = Field(None, ge=0, description="Average visibility")

    # Thunder probability (%)
    thunder_probability: Optional[float] = Field(None, ge=0, le=100, description="Thunder probability")


# ============================= MAIN SCHEMA =============================

class WeatherDataSchema(BaseModel):
    """Schema chính cho dữ liệu thời tiết."""

    # Thông tin vị trí
    location: LocationSchema = Field(..., description="Thông tin vị trí trạm quan trắc")

    # Time
    timestamp: datetime = Field(..., description="Observation timestamp")
    data_time: datetime = Field(..., description="Data update time")

    # Metadata
    data_source: DataSource = Field(..., description="Data source")
    data_quality: DataQuality = Field(..., description="Data quality")

    # Chỉ số thời tiết
    metrics: WeatherMetricsSchema = Field(..., description="Các chỉ số thời tiết")

    class Config:
        """Cấu hình Pydantic."""
        allow_population_by_field_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

    @validator('timestamp', 'data_time', pre=True, check_fields=False)
    def parse_datetime(cls, v):
        """Parse datetime từ string nếu cần."""
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v

    @validator('data_source', pre=True, check_fields=False)
    def parse_data_source(cls, v):
        """Parse data source từ string."""
        if isinstance(v, str):
            return DataSource(v.lower())
        return v

    @validator('data_quality', pre=True, check_fields=False)
    def parse_data_quality(cls, v):
        """Parse data quality từ string."""
        if isinstance(v, str):
            return DataQuality(v.lower())
        return v

    def to_flat_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thành dict phẳng để tương thích với DataFrame."""
        data = self.dict()
        location = data.pop('location')
        metrics = data.pop('metrics')

        # Flatten location
        for key, value in location.items():
            data[f'location_{key}'] = value

        # Flatten metrics
        for key, value in metrics.items():
            data[key] = value

        return data

    @classmethod
    def from_flat_dict(cls, data: Dict[str, Any]) -> WeatherDataSchema:
        """Tạo instance từ dict phẳng với tên cột snake_case tiếng Anh."""
        # Không mutate dict gốc
        data = dict(data)

        # Extract location fields (snake_case English)
        location_fields = ['station_id', 'station_name', 'province', 'district', 'latitude', 'longitude']
        location_data = {}
        for f in location_fields:
            prefixed = f'location_{f}'
            if prefixed in data:
                location_data[f] = data.pop(prefixed)
            elif f in data:
                location_data[f] = data.pop(f)

        # Extract top-level fields
        top_level_keys = ['timestamp', 'data_time', 'data_source', 'data_quality']
        top_level = {k: data.pop(k) for k in top_level_keys if k in data}

        # Còn lại là metrics
        metrics_data = data

        return cls(
            location=LocationSchema(**location_data),
            metrics=WeatherMetricsSchema(**metrics_data),
            **top_level,
        )


# ============================= UTILITY FUNCTIONS =============================

def validate_weather_data(data: Dict[str, Any]) -> WeatherDataSchema:
    """Validate một bản ghi dữ liệu thời tiết."""
    return WeatherDataSchema(**data)


def validate_weather_dataframe(df: pd.DataFrame) -> List[WeatherDataSchema]:
    """Validate toàn bộ DataFrame."""
    records = []
    for _, row in df.iterrows():
        try:
            record = WeatherDataSchema.from_flat_dict(row.to_dict())
            records.append(record)
        except Exception as e:
            # Log lỗi và bỏ qua bản ghi không hợp lệ
            logger.warning("Invalid record: %s", e)
            continue
    return records


def get_schema_fields() -> Dict[str, Dict[str, Any]]:
    """Lấy thông tin các trường trong schema."""
    schema = WeatherDataSchema.schema()
    return schema['properties']