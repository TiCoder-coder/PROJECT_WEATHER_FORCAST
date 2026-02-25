"""
paths.py - Centralized path configuration cho toàn bộ project.

Tất cả đường dẫn được tính DYNAMIC dựa trên vị trí thực tế của project,
KHÔNG hardcode đường dẫn Linux/Windows.

Sử dụng:
    from Weather_Forcast_App.paths import DATA_CRAWL_DIR, DATA_MERGE_DIR, ...

Hoặc trong scripts chạy trực tiếp:
    import sys
    from pathlib import Path
    _here = Path(__file__).resolve()
    _project_root = _here.parent.parent  # adjust as needed
    sys.path.insert(0, str(_project_root))
    from Weather_Forcast_App.paths import *
"""
from pathlib import Path

# ============================================================
# PROJECT ROOT - tự động detect từ vị trí file này
# ============================================================
# File này nằm ở: PROJECT_ROOT/Weather_Forcast_App/paths.py
# => project root = parent of Weather_Forcast_App
_THIS_FILE = Path(__file__).resolve()
APP_ROOT = _THIS_FILE.parent                    # Weather_Forcast_App/
PROJECT_ROOT = _THIS_FILE.parent.parent          # PROJECT_ROOT/

# ============================================================
# DATA DIRECTORIES
# ============================================================
DATA_ROOT = PROJECT_ROOT / "data"
DATA_CRAWL_DIR = DATA_ROOT / "data_crawl"
DATA_MERGE_DIR = DATA_ROOT / "data_merge"
DATA_CLEAN_ROOT = DATA_ROOT / "data_clean"
DATA_CLEAN_MERGE_DIR = DATA_CLEAN_ROOT / "data_merge_clean"
DATA_CLEAN_NOT_MERGE_DIR = DATA_CLEAN_ROOT / "data_not_merge_clean"

# ============================================================
# SCRIPT PATHS
# ============================================================
SCRIPTS_DIR = APP_ROOT / "scripts"
SCRIPT_CRAWL_BY_API = SCRIPTS_DIR / "Crawl_data_by_API.py"
SCRIPT_CRAWL_VRAIN_HTML = SCRIPTS_DIR / "Crawl_data_from_html_of_Vrain.py"
SCRIPT_CRAWL_VRAIN_API = SCRIPTS_DIR / "Crawl_data_from_Vrain_by_API.py"
SCRIPT_CRAWL_VRAIN_SELENIUM = SCRIPTS_DIR / "Crawl_data_from_Vrain_by_Selenium.py"
SCRIPT_MERGE_XLSX = SCRIPTS_DIR / "Merge_xlsx.py"
SCRIPT_CLEARDATA = SCRIPTS_DIR / "Cleardata.py"

# ============================================================
# ML MODEL DIRECTORIES
# ============================================================
ML_MODEL_ROOT = APP_ROOT / "Machine_learning_model"
ML_DATASET_AFTER_SPLIT = ML_MODEL_ROOT / "Dataset_after_split"
ML_ARTIFACTS_LATEST = APP_ROOT / "Machine_learning_artifacts" / "latest"

# ============================================================
# DATABASE
# ============================================================
DB_PATH = PROJECT_ROOT / "vietnam_weather.db"

# ============================================================
# UTILITY: Ensure directories exist
# ============================================================
def ensure_data_dirs():
    """Tạo tất cả thư mục data nếu chưa tồn tại."""
    for d in [DATA_CRAWL_DIR, DATA_MERGE_DIR, DATA_CLEAN_MERGE_DIR, 
              DATA_CLEAN_NOT_MERGE_DIR, ML_ARTIFACTS_LATEST]:
        d.mkdir(parents=True, exist_ok=True)
