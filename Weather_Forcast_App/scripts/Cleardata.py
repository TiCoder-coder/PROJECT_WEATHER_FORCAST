import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io 
import os
import json

from django.http import JsonResponse
from django.conf import settings
from datetime import datetime
from sklearn.impute import SimpleImputer


def clean_data_view(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'message': 'Method not allowed'}, status=405)

    try:
        data = json.loads(request.body)
        filename = data.get('filename')
        file_type = (data.get("file_type") or "merged").lower()
        action = data.get('action', 'analyze')        # analyze | clean

        if file_type == 'merged':
            file_path = os.path.join(
                settings.BASE_DIR,
                'Weather_Forcast_App', 'Merge_data', filename
            )
        else:
            file_path = os.path.join(
                settings.BASE_DIR,
                'Weather_Forcast_App', 'output', filename
            )

        if not os.path.exists(file_path):
            return JsonResponse({'success': False, 'message': 'File không tồn tại'})

        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".xlsx", ".xls"]:
            data_df = pd.read_excel(file_path, engine="openpyxl")
        elif ext == ".csv":
            try:
                data_df = pd.read_csv(file_path, encoding="utf-8")
            except UnicodeDecodeError:
                data_df = pd.read_csv(file_path, encoding="utf-8-sig")
        else:
            return JsonResponse({'success': False, 'message': 'Chỉ hỗ trợ CSV/XLSX'}, status=400)


        if action == 'analyze':
            return JsonResponse({
                'success': True,
                'analysis': analyze_missing_data(data_df, filename)
            })

        if action == 'clean':
            return JsonResponse({
                'success': True,
                **perform_cleaning(data_df, filename, file_type)
            })

    except Exception as e:
        return JsonResponse({'success': False, 'message': str(e)})
def analyze_missing_data(data_df, filename):
    total_rows = len(data_df)
    total_columns = len(data_df.columns)

    data_df.replace(["N/A", "NA", "null", ""], np.nan, inplace=True)

    missing_report = []
    for col in data_df.columns:
        missing = data_df[col].isna().sum()
        missing_report.append({
            "column": col,
            "missing_count": int(missing),
            "percent": round(missing / total_rows * 100, 2),
            "dtype": str(data_df[col].dtype)
        })

    plt.figure(figsize=(12, 8))
    sns.heatmap(data_df.isna(), cmap="Blues", yticklabels=False)
    plt.title(f"Missing Data Heatmap - {filename}")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    heatmap = base64.b64encode(buf.getvalue()).decode()

    return {
        "filename": filename,
        "total_rows": total_rows,
        "total_columns": total_columns,
        "missing_report": missing_report,
        "heatmap_image": heatmap
    }
def perform_cleaning(data_df, filename, file_type="merged"):
    original_df = data_df.copy()
    cleaning_log = {}
    file_type = (file_type or "merged").lower()

    if file_type not in ("merged", "output"):
        file_type = "merged"

    # =========================
    # Chuẩn hóa missing
    # =========================
    data_df.columns = [str(c).strip() for c in data_df.columns]
    data_df.replace(["N/A", "NA", "null", "NULL", ""], np.nan, inplace=True)

    # =========================
    # Loại dòng rác / header lặp - CHỈ LOẠI DÒNG THỰC SỰ LÀ HEADER
    # =========================
    first_col = data_df.columns[0]
    
    # CHỈ xóa dòng có chứa "DANH SÁCH" trong cột đầu tiên
    data_df = data_df[~data_df[first_col].astype(str).str.contains("DANH SÁCH", case=False, na=False)]
    
    # Kiểm tra cột "Tên Trạm" nếu có
    if "Tên Trạm" in data_df.columns:
        # CHỈ xóa dòng mà toàn bộ giá trị đều giống header (dòng header lặp)
        header_mask = data_df["Tên Trạm"].astype(str).str.strip().eq("Tên Trạm") & \
                     data_df.iloc[:, 0].astype(str).str.strip().ne(data_df.iloc[:, 0].astype(str).str.strip().iloc[0] if len(data_df) > 0 else "")
        data_df = data_df[~header_mask]

    # CHỈ xóa dòng trùng lặp nếu tất cả giá trị đều giống nhau
    duplicate_header_mask = pd.Series([False] * len(data_df))
    for col in data_df.columns:
        if data_df[col].dtype == object:
            duplicate_header_mask = duplicate_header_mask | data_df[col].astype(str).str.strip().eq(col.strip())
    
    # Chỉ xóa nếu có nhiều hơn 50% cột có giá trị trùng với tên cột
    col_threshold = len(data_df.columns) * 0.5
    rows_to_drop = duplicate_header_mask.groupby(duplicate_header_mask.index).sum() > col_threshold
    data_df = data_df[~rows_to_drop]

    # Reset index sau khi xóa
    data_df = data_df.reset_index(drop=True)

    # =========================
    # KIỂM TRA: In ra số dòng sau khi xử lý
    # =========================
    print(f"Số dòng sau khi loại header: {len(data_df)}")
    if len(data_df) > 0:
        print(f"Mẫu dữ liệu đầu tiên:\n{data_df.head(2)}")
    else:
        print("KHÔNG CÓ DỮ LIỆU SAU KHI XỬ LÝ!")

    # =========================
    # Xử lý dữ liệu âm
    # =========================
    negative_fixed = {}
    for col in data_df.select_dtypes(include=[np.number]).columns:
        count = (data_df[col] < 0).sum()
        if count > 0:
            data_df.loc[data_df[col] < 0, col] = 0
            negative_fixed[col] = int(count)
    cleaning_log["negative_fixed"] = negative_fixed

    # =========================
    # Chuẩn hóa kiểu dữ liệu
    # =========================
    dtype_log = {}
    for col in data_df.columns:
        if data_df[col].dtype == object:
            # Kiểm tra nếu cột có thể chuyển thành số
            try:
                converted = pd.to_numeric(data_df[col], errors="coerce")
                not_null_count = converted.notna().sum()
                total_count = len(data_df[col])
                ratio = not_null_count / total_count if total_count > 0 else 0
                
                if ratio >= 0.85 and not_null_count > 0:
                    data_df[col] = converted
                    dtype_log[col] = f"string → numeric (parsed {ratio:.0%})"
            except:
                pass

        col_l = col.lower()
        if ("date" in col_l) or ("time" in col_l) or ("thời gian" in col_l) or ("ngày" in col_l):
            try:
                data_df[col] = pd.to_datetime(data_df[col], errors="coerce")
                dtype_log[col] = "→ datetime"
            except:
                pass

    cleaning_log["datatype_standardized"] = dtype_log

    # =========================
    # Xử lý missing values - CHỈ áp dụng nếu có dữ liệu
    # =========================
    if len(data_df) > 0:
        num_cols = data_df.select_dtypes(include=[np.number]).columns.tolist()
        dt_cols = data_df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
        cat_cols = [c for c in data_df.columns if c not in num_cols and c not in dt_cols]

        for c in num_cols:
            if c in data_df.columns and len(data_df[c]) > 0:
                m = data_df[c].mean()
                data_df[c] = data_df[c].fillna(m if pd.notna(m) else 0)

        for c in dt_cols:
            if c in data_df.columns and len(data_df[c]) > 0:
                mode = data_df[c].mode()
                data_df[c] = data_df[c].fillna(mode.iloc[0] if not mode.empty else pd.NaT)

        for c in cat_cols:
            if c in data_df.columns and len(data_df[c]) > 0:
                mode = data_df[c].mode()
                data_df[c] = data_df[c].fillna(mode.iloc[0] if not mode.empty else "")
    else:
        # Nếu không có dữ liệu, trả về DataFrame rỗng
        return {
            "message": "Không có dữ liệu sau khi làm sạch",
            "output_file": None,
            "report_file": None
        }

    # =========================
    # So sánh trước – sau
    # =========================
    original_missing_like = original_df.replace(["N/A","NA","null","NULL",""], np.nan)
    comparison = {
        "rows_before": len(original_df),
        "rows_after": len(data_df),
        "columns_before": original_df.shape[1],
        "columns_after": data_df.shape[1],
        "missing_before": int(original_missing_like.isna().sum().sum()),
        "missing_after": int(data_df.isna().sum().sum()),
    }

    # =========================
    # Xuất file CSV - CHỈ nếu có dữ liệu
    # =========================
    if len(data_df) == 0:
        return {
            "message": "Không có dữ liệu để xuất file",
            "output_file": None,
            "report_file": None
        }

    today = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_filename = f"{os.path.splitext(filename)[0]}_cleaned_{today}.csv"

    base = os.path.join(settings.BASE_DIR, "Weather_Forcast_App", "cleaned_data")
    if file_type == "merged":
        output_dir = os.path.join(base, "Clean_Data_For_File_Merge")
    else:
        output_dir = os.path.join(base, "Clean_Data_For_File_Not_Merge")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, clean_filename)
    data_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # =========================
    # Xuất báo cáo JSON
    # =========================
    report_path = output_path.replace(".csv", "_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "comparison": comparison,
            "cleaning_log": cleaning_log,
            "sample_data": data_df.head(5).to_dict(orient="records") if len(data_df) > 0 else []
        }, f, ensure_ascii=False, indent=4)

    return {
        "message": "Làm sạch dữ liệu hoàn tất",
        "output_file": clean_filename,
        "report_file": os.path.basename(report_path),
        "rows_remaining": len(data_df)
    }
