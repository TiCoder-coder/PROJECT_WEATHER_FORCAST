# ----------------------------- DATA PACKAGE INIT -----------------------------------------------------------
"""
Weather_Forcast_App.data Package

Package này chứa các module xử lý và load dữ liệu.

Modules:
    - Loader: Module load dữ liệu từ các thư mục dataset (CSV/Excel/JSON/TXT)

Usage:
    from Weather_Forcast_App.data import DataLoader
    from Weather_Forcast_App.data import FileInfo, LoadResult
"""

from .Loader import (
    DataLoader,
    FileInfo,
    LoadResult,
    FOLDER_MAPPING,
    SUPPORTED_EXTENSIONS
)

__all__ = [
    'DataLoader',
    'FileInfo',
    'LoadResult',
    'FOLDER_MAPPING',
    'SUPPORTED_EXTENSIONS'
]
