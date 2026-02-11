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
    ma_tram: str = Field(..., description="Mã trạm quan trắc")
    ten_tram: str = Field(..., description="Tên trạm quan trắc")
    tinh_thanh_pho: str = Field(..., description="Tỉnh/Thành phố")
    huyen: str = Field(..., description="Huyện/Quận")
    vi_do: float = Field(..., ge=-90, le=90, description="Vĩ độ")
    kinh_do: float = Field(..., ge=-180, le=180, description="Kinh độ")


class WeatherMetricsSchema(BaseModel):
    """Schema cho các chỉ số thời tiết."""
    # Nhiệt độ (°C)
    nhiet_do_hien_tai: Optional[float] = Field(None, description="Nhiệt độ hiện tại")
    nhiet_do_toi_da: Optional[float] = Field(None, description="Nhiệt độ tối đa")
    nhiet_do_toi_thieu: Optional[float] = Field(None, description="Nhiệt độ tối thiểu")
    nhiet_do_trung_binh: Optional[float] = Field(None, description="Nhiệt độ trung bình")

    # Độ ẩm (%)
    do_am_hien_tai: Optional[float] = Field(None, ge=0, le=100, description="Độ ẩm hiện tại")
    do_am_toi_da: Optional[float] = Field(None, ge=0, le=100, description="Độ ẩm tối đa")
    do_am_toi_thieu: Optional[float] = Field(None, ge=0, le=100, description="Độ ẩm tối thiểu")
    do_am_trung_binh: Optional[float] = Field(None, ge=0, le=100, description="Độ ẩm trung bình")

    # Áp suất (hPa)
    ap_suat_hien_tai: Optional[float] = Field(None, description="Áp suất hiện tại")
    ap_suat_toi_da: Optional[float] = Field(None, description="Áp suất tối đa")
    ap_suat_toi_thieu: Optional[float] = Field(None, description="Áp suất tối thiểu")
    ap_suat_trung_binh: Optional[float] = Field(None, description="Áp suất trung bình")

    # Tốc độ gió (m/s)
    toc_do_gio_hien_tai: Optional[float] = Field(None, ge=0, description="Tốc độ gió hiện tại")
    toc_do_gio_toi_da: Optional[float] = Field(None, ge=0, description="Tốc độ gió tối đa")
    toc_do_gio_toi_thieu: Optional[float] = Field(None, ge=0, description="Tốc độ gió tối thiểu")
    toc_do_gio_trung_binh: Optional[float] = Field(None, ge=0, description="Tốc độ gió trung bình")

    # Hướng gió (độ)
    huong_gio_hien_tai: Optional[float] = Field(None, ge=0, le=360, description="Hướng gió hiện tại")
    huong_gio_trung_binh: Optional[float] = Field(None, ge=0, le=360, description="Hướng gió trung bình")

    # Lượng mưa (mm)
    luong_mua_hien_tai: Optional[float] = Field(None, ge=0, description="Lượng mưa hiện tại")
    luong_mua_toi_da: Optional[float] = Field(None, ge=0, description="Lượng mưa tối đa")
    luong_mua_toi_thieu: Optional[float] = Field(None, ge=0, description="Lượng mưa tối thiểu")
    luong_mua_trung_binh: Optional[float] = Field(None, ge=0, description="Lượng mưa trung bình")
    tong_luong_mua: Optional[float] = Field(None, ge=0, description="Tổng lượng mưa")

    # Độ che phủ mây (%)
    do_che_phu_may_hien_tai: Optional[float] = Field(None, ge=0, le=100, description="Độ che phủ mây hiện tại")
    do_che_phu_may_toi_da: Optional[float] = Field(None, ge=0, le=100, description="Độ che phủ mây tối đa")
    do_che_phu_may_toi_thieu: Optional[float] = Field(None, ge=0, le=100, description="Độ che phủ mây tối thiểu")
    do_che_phu_may_trung_binh: Optional[float] = Field(None, ge=0, le=100, description="Độ che phủ mây trung bình")

    # Tầm nhìn (km)
    tam_nhin_hien_tai: Optional[float] = Field(None, ge=0, description="Tầm nhìn hiện tại")
    tam_nhin_toi_da: Optional[float] = Field(None, ge=0, description="Tầm nhìn tối đa")
    tam_nhin_toi_thieu: Optional[float] = Field(None, ge=0, description="Tầm nhìn tối thiểu")
    tam_nhin_trung_binh: Optional[float] = Field(None, ge=0, description="Tầm nhìn trung bình")

    # Xác suất sấm sét (%)
    xac_suat_sam_set: Optional[float] = Field(None, ge=0, le=100, description="Xác suất sấm sét")


# ============================= MAIN SCHEMA =============================

class WeatherDataSchema(BaseModel):
    """Schema chính cho dữ liệu thời tiết."""

    # Thông tin vị trí
    location: LocationSchema = Field(..., description="Thông tin vị trí trạm quan trắc")

    # Thời gian
    dau_thoi_gian: datetime = Field(..., description="Dấu thời gian quan trắc")
    thoi_gian_cap_nhat: datetime = Field(..., description="Thời gian cập nhật dữ liệu")

    # Metadata
    nguon_du_lieu: DataSource = Field(..., description="Nguồn dữ liệu")
    chat_luong_du_lieu: DataQuality = Field(..., description="Chất lượng dữ liệu")

    # Chỉ số thời tiết
    metrics: WeatherMetricsSchema = Field(..., description="Các chỉ số thời tiết")

    class Config:
        """Cấu hình Pydantic."""
        allow_population_by_field_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

    @validator('dau_thoi_gian', 'thoi_gian_cap_nhat', pre=True)
    def parse_datetime(cls, v):
        """Parse datetime từ string nếu cần."""
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v

    @validator('nguon_du_lieu', pre=True)
    def parse_data_source(cls, v):
        """Parse data source từ string."""
        if isinstance(v, str):
            return DataSource(v.lower())
        return v

    @validator('chat_luong_du_lieu', pre=True)
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
        """Tạo instance từ dict phẳng."""
        # Không mutate dict gốc
        data = dict(data)

        # Extract location fields
        location_fields = ['ma_tram', 'ten_tram', 'tinh_thanh_pho', 'huyen', 'vi_do', 'kinh_do']
        location_data = {}
        for f in location_fields:
            prefixed = f'location_{f}'
            if prefixed in data:
                location_data[f] = data.pop(prefixed)
            elif f in data:
                location_data[f] = data.pop(f)

        # Extract top-level fields
        top_level_keys = ['dau_thoi_gian', 'thoi_gian_cap_nhat', 'nguon_du_lieu', 'chat_luong_du_lieu']
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