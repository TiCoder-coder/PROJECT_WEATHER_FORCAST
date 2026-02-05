# ----------------------------- DATA LOADER - MODULE LOAD DỮ LIỆU -----------------------------------------------------------
"""
Loader.py - Module load và xử lý dữ liệu từ các thư mục dataset

Mục đích:
    - Load dữ liệu từ các file CSV, Excel, JSON, TXT
    - Hỗ trợ pagination cho file lớn
    - Cung cấp thông tin metadata của file
    - Validate và bảo mật đường dẫn file

Thư mục dữ liệu:
    - output/           : Dữ liệu thô sau crawl (chưa xử lý)
    - Merge_data/       : Dữ liệu đã gộp (merged)
    - cleaned_data/     : Dữ liệu đã làm sạch
        - Clean_Data_For_File_Merge/     : Clean từ file đã merge
        - Clean_Data_For_File_Not_Merge/ : Clean từ file thô

Cách sử dụng:
    from Weather_Forcast_App.data import DataLoader
    
    loader = DataLoader()
    
    # Lấy danh sách file
    files = loader.list_files('output')
    
    # Load dữ liệu với pagination
    result = loader.load_file('output', 'data.csv', page=1, per_page=100)
    
    # Lấy thông tin file
    info = loader.get_file_info('merged', 'merged_data.xlsx')
"""

import os
import json
import mimetypes
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


# ============================= CONSTANTS =============================

# Mapping folder key -> đường dẫn thực tế (relative to app)
FOLDER_MAPPING: Dict[str, str] = {
    'output': 'output',
    'merged': 'Merge_data',
    'cleaned': 'cleaned_data',
    'cleaned_merge': 'cleaned_data/Clean_Data_For_File_Merge',
    'cleaned_raw': 'cleaned_data/Clean_Data_For_File_Not_Merge',
}

# Các extension được hỗ trợ
SUPPORTED_EXTENSIONS: Tuple[str, ...] = ('.csv', '.xlsx', '.xls', '.json', '.txt')

# Default pagination
DEFAULT_PAGE_SIZE = 10
MAX_PAGE_SIZE = 100000000000000


# ============================= ENUMS =============================

class FileType(Enum):
    """Loại file được hỗ trợ."""
    CSV = 'csv'
    EXCEL = 'excel'
    JSON = 'json'
    TEXT = 'text'
    UNKNOWN = 'unknown'


class LoadStatus(Enum):
    """Trạng thái load file."""
    SUCCESS = 'success'
    ERROR = 'error'
    FILE_NOT_FOUND = 'file_not_found'
    INVALID_FOLDER = 'invalid_folder'
    UNSUPPORTED_FORMAT = 'unsupported_format'
    PERMISSION_DENIED = 'permission_denied'


# ============================= DATA CLASSES =============================

@dataclass
class FileInfo:
    """
    Thông tin metadata của file.
    
    Attributes:
        name: Tên file
        folder: Folder key
        path: Đường dẫn đầy đủ
        size_bytes: Kích thước (bytes)
        size_display: Kích thước hiển thị (KB/MB)
        file_type: Loại file
        extension: Phần mở rộng
        modified_time: Thời gian sửa đổi
        created_time: Thời gian tạo
        row_count: Số dòng (nếu là CSV/Excel)
        column_count: Số cột (nếu là CSV/Excel)
        columns: Danh sách tên cột
    """
    name: str
    folder: str
    path: str
    size_bytes: int
    size_display: str
    file_type: FileType
    extension: str
    modified_time: datetime
    created_time: Optional[datetime] = None
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    columns: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang dictionary."""
        return {
            'name': self.name,
            'folder': self.folder,
            'path': self.path,
            'size_bytes': self.size_bytes,
            'size_display': self.size_display,
            'file_type': self.file_type.value,
            'extension': self.extension,
            'modified_time': self.modified_time.isoformat() if self.modified_time else None,
            'created_time': self.created_time.isoformat() if self.created_time else None,
            'row_count': self.row_count,
            'column_count': self.column_count,
            'columns': self.columns,
        }


@dataclass
class LoadResult:
    """
    Kết quả load file.
    
    Attributes:
        status: Trạng thái load
        data: Dữ liệu đã load (DataFrame hoặc dict/list/string)
        file_info: Thông tin file
        message: Thông báo (lỗi hoặc thành công)
        page: Trang hiện tại
        per_page: Số dòng mỗi trang
        total_rows: Tổng số dòng
        total_pages: Tổng số trang
        has_next: Có trang tiếp không
        has_prev: Có trang trước không
    """
    status: LoadStatus
    data: Any = None
    file_info: Optional[FileInfo] = None
    message: str = ''
    page: int = 1
    per_page: int = DEFAULT_PAGE_SIZE
    total_rows: int = 0
    total_pages: int = 0
    has_next: bool = False
    has_prev: bool = False
    
    @property
    def is_success(self) -> bool:
        """Kiểm tra load thành công."""
        return self.status == LoadStatus.SUCCESS
    
    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang dictionary."""
        data_serialized = None
        if self.data is not None:
            if isinstance(self.data, pd.DataFrame):
                data_serialized = self.data.to_dict(orient='records')
            elif isinstance(self.data, (dict, list)):
                data_serialized = self.data
            else:
                data_serialized = str(self.data)
        
        return {
            'status': self.status.value,
            'data': data_serialized,
            'file_info': self.file_info.to_dict() if self.file_info else None,
            'message': self.message,
            'pagination': {
                'page': self.page,
                'per_page': self.per_page,
                'total_rows': self.total_rows,
                'total_pages': self.total_pages,
                'has_next': self.has_next,
                'has_prev': self.has_prev,
            }
        }


# ============================= MAIN CLASS =============================

class DataLoader:
    """
    Class chính để load và xử lý dữ liệu từ các thư mục dataset.
    
    Tính năng:
        - Load CSV, Excel, JSON, TXT files
        - Pagination cho file lớn
        - Caching metadata
        - Validate đường dẫn (bảo mật)
        - Thống kê file
    
    Example:
        >>> loader = DataLoader()
        >>> 
        >>> # Lấy danh sách file trong thư mục output
        >>> files = loader.list_files('output')
        >>> for f in files:
        ...     print(f.name, f.size_display)
        >>> 
        >>> # Load file với pagination
        >>> result = loader.load_file('merged', 'data.csv', page=1, per_page=50)
        >>> if result.is_success:
        ...     df = result.data
        ...     print(f"Loaded {len(df)} rows")
        >>> 
        >>> # Lấy thông tin chi tiết file
        >>> info = loader.get_file_info('cleaned_merge', 'cleaned_data.csv')
        >>> print(f"Rows: {info.row_count}, Columns: {info.columns}")
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Khởi tạo DataLoader.
        
        Args:
            base_path: Đường dẫn gốc đến thư mục app.
                       Nếu None, tự động detect từ vị trí file này.
        """
        if base_path:
            self.base_path = Path(base_path)
        else:
            self.base_path = Path(__file__).parent.parent
        
        self._file_cache: Dict[str, FileInfo] = {}
    
    # ============================= PUBLIC METHODS =============================
    
    def get_folder_path(self, folder_key: str) -> Optional[Path]:
        """
        Lấy đường dẫn thực tế của folder từ folder key.
        
        Args:
            folder_key: Key của folder (output, merged, cleaned, ...)
            
        Returns:
            Path object nếu folder hợp lệ, None nếu không
            
        Example:
            >>> loader.get_folder_path('output')
            PosixPath('/path/to/Weather_Forcast_App/output')
        """
        if folder_key not in FOLDER_MAPPING:
            return None
        
        relative_path = FOLDER_MAPPING[folder_key]
        full_path = self.base_path / relative_path
        
        return full_path if full_path.exists() else None
    
    def list_files(
        self, 
        folder_key: str, 
        extensions: Optional[Tuple[str, ...]] = None,
        sort_by: str = 'modified',
        reverse: bool = True
    ) -> List[FileInfo]:
        """
        Liệt kê tất cả các file trong folder.
        
        Args:
            folder_key: Key của folder
            extensions: Tuple các extension cần lọc. None = tất cả supported
            sort_by: Sắp xếp theo 'name', 'modified', 'size'
            reverse: True = giảm dần, False = tăng dần
            
        Returns:
            List các FileInfo objects
            
        Example:
            >>> files = loader.list_files('output', extensions=('.csv',))
            >>> for f in files:
            ...     print(f"{f.name}: {f.size_display}")
        """
        folder_path = self.get_folder_path(folder_key)
        if not folder_path:
            return []
        
        extensions = extensions or SUPPORTED_EXTENSIONS
        files: List[FileInfo] = []
        
        try:
            for item in folder_path.iterdir():
                if item.is_file() and item.suffix.lower() in extensions:
                    file_info = self._get_basic_file_info(item, folder_key)
                    if file_info:
                        files.append(file_info)
        except PermissionError:
            return []
        
        # Sort
        sort_key_map = {
            'name': lambda x: x.name.lower(),
            'modified': lambda x: x.modified_time,
            'size': lambda x: x.size_bytes,
        }
        sort_key = sort_key_map.get(sort_by, sort_key_map['modified'])
        files.sort(key=sort_key, reverse=reverse)
        
        return files
    
    def get_file_info(
        self, 
        folder_key: str, 
        filename: str,
        include_data_stats: bool = True
    ) -> Optional[FileInfo]:
        """
        Lấy thông tin chi tiết của file.
        
        Args:
            folder_key: Key của folder
            filename: Tên file
            include_data_stats: True = đọc file để lấy row_count, columns
            
        Returns:
            FileInfo object hoặc None nếu file không tồn tại
            
        Example:
            >>> info = loader.get_file_info('merged', 'data.csv')
            >>> print(f"Rows: {info.row_count}, Cols: {info.column_count}")
        """
        file_path = self._validate_and_get_path(folder_key, filename)
        if not file_path:
            return None
        
        # Check cache
        cache_key = str(file_path)
        if cache_key in self._file_cache:
            cached = self._file_cache[cache_key]
            if file_path.stat().st_mtime == cached.modified_time.timestamp():
                return cached
        
        file_info = self._get_basic_file_info(file_path, folder_key)
        if not file_info:
            return None
        
        # Lấy thêm data stats nếu cần
        if include_data_stats and file_info.file_type in (FileType.CSV, FileType.EXCEL):
            try:
                df = self._read_dataframe(file_path, nrows=0)
                file_info.columns = list(df.columns)
                file_info.column_count = len(df.columns)
                
                # Đếm số dòng
                if file_info.file_type == FileType.CSV:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        file_info.row_count = sum(1 for _ in f) - 1  # Trừ header
                else:
                    df_full = pd.read_excel(file_path)
                    file_info.row_count = len(df_full)
            except Exception:
                pass
        
        # Cache
        self._file_cache[cache_key] = file_info
        return file_info
    
    def load_file(
        self,
        folder_key: str,
        filename: str,
        page: int = 1,
        per_page: int = DEFAULT_PAGE_SIZE,
        columns: Optional[List[str]] = None
    ) -> LoadResult:
        """
        Load dữ liệu từ file với pagination.
        
        Args:
            folder_key: Key của folder
            filename: Tên file
            page: Số trang (1-indexed)
            per_page: Số dòng mỗi trang
            columns: List các cột cần load (None = tất cả)
            
        Returns:
            LoadResult object chứa dữ liệu và metadata
            
        Example:
            >>> result = loader.load_file('output', 'data.csv', page=2, per_page=50)
            >>> if result.is_success:
            ...     df = result.data
            ...     print(f"Page {result.page}/{result.total_pages}")
            ...     print(f"Has next: {result.has_next}")
        """
        # Validate folder
        if folder_key not in FOLDER_MAPPING:
            return LoadResult(
                status=LoadStatus.INVALID_FOLDER,
                message=f"Invalid folder key: {folder_key}. Valid keys: {list(FOLDER_MAPPING.keys())}"
            )
        
        # Validate và lấy path
        file_path = self._validate_and_get_path(folder_key, filename)
        if not file_path:
            return LoadResult(
                status=LoadStatus.FILE_NOT_FOUND,
                message=f"File not found: {filename} in {folder_key}"
            )
        
        # Lấy file info
        file_info = self.get_file_info(folder_key, filename, include_data_stats=True)
        
        # Validate extension
        file_type = self._get_file_type(file_path)
        if file_type == FileType.UNKNOWN:
            return LoadResult(
                status=LoadStatus.UNSUPPORTED_FORMAT,
                message=f"Unsupported file format: {file_path.suffix}",
                file_info=file_info
            )
        
        # Validate pagination params
        page = max(1, page)
        per_page = min(max(1, per_page), MAX_PAGE_SIZE)
        
        try:
            # Load theo loại file
            if file_type in (FileType.CSV, FileType.EXCEL):
                return self._load_tabular(file_path, file_info, page, per_page, columns)
            elif file_type == FileType.JSON:
                return self._load_json(file_path, file_info)
            elif file_type == FileType.TEXT:
                return self._load_text(file_path, file_info, page, per_page)
            else:
                return LoadResult(
                    status=LoadStatus.UNSUPPORTED_FORMAT,
                    message=f"Cannot load file type: {file_type.value}",
                    file_info=file_info
                )
        except PermissionError:
            return LoadResult(
                status=LoadStatus.PERMISSION_DENIED,
                message=f"Permission denied: Cannot read {filename}",
                file_info=file_info
            )
        except Exception as e:
            return LoadResult(
                status=LoadStatus.ERROR,
                message=f"Error loading file: {str(e)}",
                file_info=file_info
            )
    
    def load_all(
        self,
        folder_key: str,
        filename: str,
        columns: Optional[List[str]] = None
    ) -> LoadResult:
        """
        Load toàn bộ file (không pagination).
        
        ⚠️ Cẩn thận với file lớn! Chỉ dùng khi cần thiết.
        
        Args:
            folder_key: Key của folder
            filename: Tên file
            columns: List các cột cần load
            
        Returns:
            LoadResult với toàn bộ dữ liệu
        """
        return self.load_file(folder_key, filename, page=1, per_page=MAX_PAGE_SIZE * 10, columns=columns)
    
    def get_preview(
        self,
        folder_key: str,
        filename: str,
        rows: int = 10
    ) -> LoadResult:
        """
        Lấy preview nhanh của file (N dòng đầu).
        
        Args:
            folder_key: Key của folder
            filename: Tên file
            rows: Số dòng preview
            
        Returns:
            LoadResult với dữ liệu preview
        """
        return self.load_file(folder_key, filename, page=1, per_page=rows)
    
    def search_files(
        self,
        query: str,
        folder_keys: Optional[List[str]] = None
    ) -> List[FileInfo]:
        """
        Tìm kiếm file theo tên.
        
        Args:
            query: Từ khóa tìm kiếm (case-insensitive)
            folder_keys: List folder cần tìm. None = tất cả
            
        Returns:
            List các FileInfo khớp với query
            
        Example:
            >>> results = loader.search_files('weather', ['output', 'merged'])
            >>> for f in results:
            ...     print(f"{f.folder}/{f.name}")
        """
        folder_keys = folder_keys or list(FOLDER_MAPPING.keys())
        query_lower = query.lower()
        results: List[FileInfo] = []
        
        for folder_key in folder_keys:
            files = self.list_files(folder_key)
            for file_info in files:
                if query_lower in file_info.name.lower():
                    results.append(file_info)
        
        return results
    
    def get_folder_stats(self, folder_key: str) -> Dict[str, Any]:
        """
        Lấy thống kê của folder.
        
        Args:
            folder_key: Key của folder
            
        Returns:
            Dict chứa thống kê: total_files, total_size, file_types, ...
            
        Example:
            >>> stats = loader.get_folder_stats('output')
            >>> print(f"Total: {stats['total_files']} files, {stats['total_size_display']}")
        """
        files = self.list_files(folder_key)
        
        total_size = sum(f.size_bytes for f in files)
        file_types: Dict[str, int] = {}
        
        for f in files:
            ft = f.file_type.value
            file_types[ft] = file_types.get(ft, 0) + 1
        
        return {
            'folder_key': folder_key,
            'folder_path': str(self.get_folder_path(folder_key)),
            'total_files': len(files),
            'total_size_bytes': total_size,
            'total_size_display': self._format_size(total_size),
            'file_types': file_types,
            'latest_file': files[0].name if files else None,
            'oldest_file': files[-1].name if files else None,
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Lấy thống kê của tất cả các folder.
        
        Returns:
            Dict với key là folder_key, value là stats
        """
        return {
            folder_key: self.get_folder_stats(folder_key)
            for folder_key in FOLDER_MAPPING.keys()
        }
    
    def clear_cache(self):
        """Xóa cache metadata."""
        self._file_cache.clear()
    
    # ============================= PRIVATE METHODS =============================
    
    def _validate_and_get_path(self, folder_key: str, filename: str) -> Optional[Path]:
        """
        Validate và trả về đường dẫn an toàn.
        
        Bảo mật:
            - Chặn path traversal (../)
            - Chỉ cho phép file trong folder được mapping
        """
        folder_path = self.get_folder_path(folder_key)
        if not folder_path:
            return None
        
        # Sanitize filename - chặn path traversal
        safe_filename = os.path.basename(filename)
        if safe_filename != filename:
            return None  # Có chứa path separator
        
        file_path = folder_path / safe_filename
        
        try:
            file_path.resolve().relative_to(folder_path.resolve())
        except ValueError:
            return None  # Path traversal detected
        
        if not file_path.exists() or not file_path.is_file():
            return None
        
        return file_path
    
    def _get_file_type(self, file_path: Path) -> FileType:
        """Xác định loại file từ extension."""
        ext = file_path.suffix.lower()
        
        if ext == '.csv':
            return FileType.CSV
        elif ext in ('.xlsx', '.xls'):
            return FileType.EXCEL
        elif ext == '.json':
            return FileType.JSON
        elif ext == '.txt':
            return FileType.TEXT
        else:
            return FileType.UNKNOWN
    
    def _get_basic_file_info(self, file_path: Path, folder_key: str) -> Optional[FileInfo]:
        """Lấy thông tin cơ bản của file (không đọc nội dung)."""
        try:
            stat = file_path.stat()
            
            return FileInfo(
                name=file_path.name,
                folder=folder_key,
                path=str(file_path),
                size_bytes=stat.st_size,
                size_display=self._format_size(stat.st_size),
                file_type=self._get_file_type(file_path),
                extension=file_path.suffix.lower(),
                modified_time=datetime.fromtimestamp(stat.st_mtime),
                created_time=datetime.fromtimestamp(stat.st_ctime),
            )
        except Exception:
            return None
    
    def _format_size(self, size_bytes: int) -> str:
        """Format kích thước file dạng KB/MB/GB."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.2f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
    
    def _read_dataframe(
        self, 
        file_path: Path, 
        nrows: Optional[int] = None,
        skiprows: Optional[int] = None,
        usecols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Đọc file thành DataFrame."""
        file_type = self._get_file_type(file_path)
        
        kwargs: Dict[str, Any] = {}
        if nrows is not None:
            kwargs['nrows'] = nrows if nrows > 0 else None
        if skiprows:
            kwargs['skiprows'] = range(1, skiprows + 1)  # Skip rows sau header
        if usecols:
            kwargs['usecols'] = usecols
        
        if file_type == FileType.CSV:
            return pd.read_csv(file_path, encoding='utf-8', **kwargs)
        elif file_type == FileType.EXCEL:
            # Excel không hỗ trợ skiprows range tốt
            if 'skiprows' in kwargs:
                del kwargs['skiprows']
                df = pd.read_excel(file_path, **kwargs)
                if skiprows:
                    return df.iloc[skiprows:]
                return df
            return pd.read_excel(file_path, **kwargs)
        else:
            raise ValueError(f"Cannot read as DataFrame: {file_type}")
    
    def _load_tabular(
        self,
        file_path: Path,
        file_info: Optional[FileInfo],
        page: int,
        per_page: int,
        columns: Optional[List[str]]
    ) -> LoadResult:
        """Load file CSV/Excel với pagination."""
        # Đếm tổng số dòng
        total_rows = file_info.row_count if file_info and file_info.row_count else 0
        
        if total_rows == 0:
            # Đếm lại nếu chưa có
            if self._get_file_type(file_path) == FileType.CSV:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    total_rows = sum(1 for _ in f) - 1
            else:
                df_count = pd.read_excel(file_path)
                total_rows = len(df_count)
        
        total_pages = (total_rows + per_page - 1) // per_page if total_rows > 0 else 1
        page = min(page, total_pages)
        
        # Calculate skip rows
        skip_rows = (page - 1) * per_page
        
        # Read data
        df = self._read_dataframe(
            file_path, 
            nrows=per_page, 
            skiprows=skip_rows if skip_rows > 0 else None,
            usecols=columns
        )
        
        return LoadResult(
            status=LoadStatus.SUCCESS,
            data=df,
            file_info=file_info,
            message=f"Loaded page {page}/{total_pages}",
            page=page,
            per_page=per_page,
            total_rows=total_rows,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
        )
    
    def _load_json(self, file_path: Path, file_info: Optional[FileInfo]) -> LoadResult:
        """Load file JSON."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Nếu là list, có thể pagination
        if isinstance(data, list):
            total_rows = len(data)
        else:
            total_rows = 1
        
        return LoadResult(
            status=LoadStatus.SUCCESS,
            data=data,
            file_info=file_info,
            message="JSON loaded successfully",
            total_rows=total_rows,
            total_pages=1,
        )
    
    def _load_text(
        self,
        file_path: Path,
        file_info: Optional[FileInfo],
        page: int,
        per_page: int
    ) -> LoadResult:
        """Load file TXT với pagination theo dòng."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        total_rows = len(lines)
        total_pages = (total_rows + per_page - 1) // per_page if total_rows > 0 else 1
        page = min(page, total_pages)
        
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_lines = lines[start_idx:end_idx]
        
        return LoadResult(
            status=LoadStatus.SUCCESS,
            data=''.join(page_lines),
            file_info=file_info,
            message=f"Loaded lines {start_idx + 1}-{min(end_idx, total_rows)} of {total_rows}",
            page=page,
            per_page=per_page,
            total_rows=total_rows,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
        )


# ============================= UTILITY FUNCTIONS =============================

def get_loader() -> DataLoader:
    """
    Factory function để lấy DataLoader instance.
    
    Returns:
        DataLoader instance với default settings
    """
    return DataLoader()


def quick_load(folder_key: str, filename: str) -> Optional[pd.DataFrame]:
    """
    Quick helper để load file thành DataFrame.
    
    Args:
        folder_key: Key của folder
        filename: Tên file
        
    Returns:
        DataFrame nếu thành công, None nếu lỗi
        
    Example:
        >>> df = quick_load('output', 'data.csv')
        >>> if df is not None:
        ...     print(df.head())
    """
    loader = get_loader()
    result = loader.load_all(folder_key, filename)
    
    if result.is_success and isinstance(result.data, pd.DataFrame):
        return result.data
    return None


def quick_list(folder_key: str) -> List[str]:
    """
    Quick helper để lấy danh sách tên file.
    
    Args:
        folder_key: Key của folder
        
    Returns:
        List tên file
    """
    loader = get_loader()
    files = loader.list_files(folder_key)
    return [f.name for f in files]