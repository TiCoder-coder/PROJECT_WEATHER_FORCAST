import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Dùng backend "Agg" (không cần giao diện) để vẽ biểu đồ trên server (Django) an toàn
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io 
import os
import json

from django.http import JsonResponse
from django.conf import settings
from datetime import datetime
from sklearn.impute import SimpleImputer  # (Hiện chưa dùng trong code này) - thường dùng để điền missing theo chiến lược (mean/median/most_frequent)


def clean_data_view(request):
    # ============================================================
    # API VIEW (Django) - XỬ LÝ "ANALYZE" HOẶC "CLEAN" DATA
    # ============================================================
    # Mục tiêu:
    # - Nhận request JSON từ frontend
    # - Đọc file dữ liệu (Excel/CSV) trong project
    # - Nếu action = analyze -> trả về thống kê missing + heatmap (base64)
    # - Nếu action = clean   -> làm sạch + xuất CSV cleaned + xuất JSON report
    #
    # Dự kiến JSON body từ client:
    # {
    #   "filename": "abc.xlsx",
    #   "file_type": "merged" | "output",
    #   "action": "analyze" | "clean"
    # }
    # ============================================================

    # Chỉ cho phép post
    # - Tránh người dùng GET vào endpoint gây lỗi
    # - Nếu không phải POST -> trả HTTP 405
    if request.method != 'POST':
        return JsonResponse({'success': False, 'message': 'Method not allowed'}, status=405)

    # Đọc body json và lấy tham số
    try:
        # request.body là bytes -> json.loads để parse thành dict
        data = json.loads(request.body)

        # filename: tên file người dùng muốn analyze/clean (vd: merged_vrain_data.xlsx)
        filename = data.get('filename')

        # file_type: phân loại đường dẫn file
        # - merged: nằm trong Weather_Forcast_App/Merge_data
        # - output: nằm trong Weather_Forcast_App/output
        # Nếu client không gửi -> mặc định merged
        file_type = (data.get("file_type") or "merged").lower()

        # action:
        # - analyze: chỉ phân tích missing, không ghi file
        # - clean: thực hiện làm sạch + xuất file + tạo report
        action = data.get('action', 'analyze')

        # ============================================================
        # XÁC ĐỊNH FILE PATH THEO file_type
        # ============================================================
        # settings.BASE_DIR: root Django project (thường là thư mục manage.py)
        # Sau đó nối với path con theo cấu trúc project của bạn.
        # Dynamic path: tự tính từ vị trí project root
        _project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        if file_type == 'merged':
            file_path = os.path.join(
                _project_root_dir, 'data', 'data_merge', filename
            )
        else:
            file_path = os.path.join(
                _project_root_dir, 'data', 'data_crawl', filename
            )

        # Kiểm tra xem file đó có đang tồn tại hay không
        # - Nếu không có file -> trả JSON báo lỗi (không ném exception)
        if not os.path.exists(file_path):
            return JsonResponse({'success': False, 'message': 'File không tồn tại'})

        # ============================================================
        # ĐỌC FILE THEO ĐUÔI (xlsx/xls/csv)
        # ============================================================
        ext = os.path.splitext(file_path)[1].lower()  # lấy phần đuôi file: ".xlsx", ".csv", ...
        # Đọc file .xlsx và .xls
        if ext in [".xlsx", ".xls"]:
            # engine="openpyxl" để đảm bảo đọc xlsx ổn định trên môi trường server
            data_df = pd.read_excel(file_path, engine="openpyxl")

        # Đọc file .csv
        elif ext == ".csv":
            # Thử 2 cách đọc csv khac nhau:
            # - utf-8: phổ biến
            # - utf-8-sig: file có BOM (thường gặp khi xuất từ Excel)
            try:
                data_df = pd.read_csv(file_path, encoding="utf-8")
            except UnicodeDecodeError:
                data_df = pd.read_csv(file_path, encoding="utf-8-sig")
        else:
            # Nếu đuôi không nằm trong danh sách hỗ trợ -> báo lỗi 400
            return JsonResponse({'success': False, 'message': 'Chỉ hỗ trợ CSV/XLSX'}, status=400)


        # ============================================================
        # ROUTE THEO action
        # ============================================================

        # action == 'analyze':
        # - Trả về phân tích missing và heatmap image (base64)
        if action == 'analyze':
            return JsonResponse({
                'success': True,
                'analysis': analyze_missing_data(data_df, filename)
            })

        # action == 'clean':
        # - Thực hiện cleaning
        # - perform_cleaning trả về dict chứa message + output_file + report_file ...
        if action == 'clean':
            return JsonResponse({
                'success': True,
                **perform_cleaning(data_df, filename, file_type)
            })

    except Exception as e:
        # Nếu có lỗi parse JSON, đọc file, hoặc lỗi runtime khác
        # -> trả message dạng string để frontend hiển thị
        return JsonResponse({'success': False, 'message': str(e)})
    
# Phân tích các lỗi missing trong file 
def analyze_missing_data(data_df, filename):
    # ============================================================
    # PHÂN TÍCH THIẾU DỮ LIỆU (MISSING DATA ANALYSIS)
    # ============================================================
    # Mục tiêu:
    # - Thống kê tổng số dòng/cột
    # - Chuẩn hoá các giá trị "giống missing" thành NaN
    # - Tạo missing_report cho từng cột: missing_count, %, dtype
    # - Vẽ heatmap missing bằng seaborn rồi encode base64 để trả về JSON
    # ============================================================

    total_rows = len(data_df) # Tính tổng số hàng
    total_columns = len(data_df.columns) # Tính tổng số cột

    # Các dạng missing
    # - Một số file có thể ghi thiếu dữ liệu dưới dạng chuỗi: "N/A", "null", "", ...
    # - replace(...) biến chúng thành np.nan để pandas hiểu là missing thật sự
    data_df.replace(["N/A", "NA", "null", ""], np.nan, inplace=True)

    # Tạo các báo cáo missing theo cột
    missing_report = []
    for col in data_df.columns:
        # isna().sum() đếm số ô bị NaN trong cột đó
        missing = data_df[col].isna().sum()

        # Với mỗi cột missing sẽ tạo báo cáo với những thành phần sau
        missing_report.append({
            "column": col, # Tên cột
            "missing_count": int(missing), # Đếm số giá trị bị missing
            "percent": round(missing / total_rows * 100, 2), # % missing trên tổng dòng
            "dtype": str(data_df[col].dtype) # dtype (kiểu dữ liệu) hiện tại của cột
        })

     # Vẽ heat map missing
    # - data_df.isna() -> DataFrame boolean (True nếu ô missing)
    # - heatmap giúp nhìn nhanh khu vực nào thiếu nhiều
    plt.figure(figsize=(12, 8))
    sns.heatmap(data_df.isna(), cmap="Blues", yticklabels=False)
    plt.title(f"Missing Data Heatmap - {filename}")
    plt.tight_layout()

    # Chuyển plot -> PNG bytes trong memory -> base64 string để trả qua JSON
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()  # đóng figure để tránh leak memory khi gọi API nhiều lần
    heatmap = base64.b64encode(buf.getvalue()).decode()

    return {
        "filename": filename,
        "total_rows": total_rows,
        "total_columns": total_columns,
        "missing_report": missing_report,
        "heatmap_image": heatmap  # base64 PNG (frontend decode để hiển thị ảnh)
    }

# Hàm dùng để cleaning data
def perform_cleaning(data_df, filename, file_type="merged"):
    # ============================================================
    # LÀM SẠCH DỮ LIỆU + XUẤT FILE CLEANED + XUẤT REPORT JSON
    # ============================================================
    # Mục tiêu:
    # - Loại các dòng rác / header lặp (nếu file bị lẫn header ở giữa)
    # - Chuẩn hoá missing
    # - Fix dữ liệu âm ở các cột số (set về 0)
    # - Chuẩn hoá dtype (cố gắng parse numeric và datetime)
    # - Fill missing:
    #   + numeric: fill mean
    #   + datetime: fill mode
    #   + categorical: fill mode
    # - So sánh trước-sau (rows/cols/missing)
    # - Lưu CSV cleaned vào thư mục phù hợp theo file_type
    # - Lưu report JSON (comparison + cleaning_log + sample)
    # ============================================================

    original_df = data_df.copy()  # lưu bản gốc để so sánh trước-sau
    cleaning_log = {}             # log chi tiết các bước cleaning (để xuất report)
    file_type = (file_type or "merged").lower()

    # Chỉ cho phép 2 loại: merged hoặc output
    # Nếu client gửi linh tinh -> ép về merged để tránh sai đường dẫn
    if file_type not in ("merged", "output"):
        file_type = "merged"

    # =========================
    # Chuẩn hóa missing
    # =========================
    # - strip tên cột để tránh lỗi "Tên Trạm " (dính space)
    data_df.columns = [str(c).strip() for c in data_df.columns]

    # - chuẩn hoá giá trị kiểu missing thành np.nan
    data_df.replace(["N/A", "NA", "null", "NULL", ""], np.nan, inplace=True)

    # =========================
    # Loại dòng rác / header lặp - CHỈ LOẠI DÒNG THỰC SỰ LÀ HEADER
    # =========================
    # - Một số file có thể có dòng mô tả như "DANH SÁCH ..." ở đầu hoặc xen giữa
    # - Một số file bị lặp header trong dữ liệu (vd: mỗi lần nối file lại chèn header)
    first_col = data_df.columns[0]
    
    # CHỈ xóa dòng có chứa "DANH SÁCH" trong cột đầu tiên
    # - case=False: không phân biệt hoa thường
    # - na=False: nếu NaN thì coi như không chứa
    data_df = data_df[~data_df[first_col].astype(str).str.contains("DANH SÁCH", case=False, na=False)]
    
    # Kiểm tra cột "Tên Trạm" nếu có
    if "Tên Trạm" in data_df.columns:
        # CHỈ xóa dòng mà toàn bộ giá trị đều giống header (dòng header lặp)
        # Ý tưởng:
        # - Nếu dòng nào có "Tên Trạm" đúng bằng chuỗi "Tên Trạm" => nghi là header lặp
        # - Đồng thời check thêm cột đầu tiên khác dòng đầu để tránh xoá nhầm
        header_mask = data_df["Tên Trạm"].astype(str).str.strip().eq("Tên Trạm") & \
                     data_df.iloc[:, 0].astype(str).str.strip().ne(data_df.iloc[:, 0].astype(str).str.strip().iloc[0] if len(data_df) > 0 else "")
        data_df = data_df[~header_mask]

    # CHỈ xóa dòng trùng lặp nếu tất cả giá trị đều giống nhau
    # Ở đây bạn tạo mask "duplicate_header_mask" theo từng cột dạng object:
    # - Nếu ô có giá trị trùng với tên cột -> có khả năng dòng đó là header lặp
    duplicate_header_mask = pd.Series([False] * len(data_df))
    for col in data_df.columns:
        if data_df[col].dtype == object:
            duplicate_header_mask = duplicate_header_mask | data_df[col].astype(str).str.strip().eq(col.strip())
    
    # Chỉ xóa nếu có nhiều hơn 50% cột có giá trị trùng với tên cột
    # - col_threshold = 50% số cột
    # - rows_to_drop: đánh dấu dòng nào thoả điều kiện bị drop
    col_threshold = len(data_df.columns) * 0.5
    rows_to_drop = duplicate_header_mask.groupby(duplicate_header_mask.index).sum() > col_threshold
    data_df = data_df[~rows_to_drop]

    # Reset index sau khi xóa
    # - tránh index bị nhảy (vd: còn 0,2,5...) gây khó debug/ghi file
    data_df = data_df.reset_index(drop=True)

    # =========================
    # KIỂM TRA: In ra số dòng sau khi xử lý
    # =========================
    # - Phần này giúp debug trực tiếp ở console server
    print(f"Số dòng sau khi loại header: {len(data_df)}")
    if len(data_df) > 0:
        print(f"Mẫu dữ liệu đầu tiên:\n{data_df.head(2)}")
    else:
        print("KHÔNG CÓ DỮ LIỆU SAU KHI XỬ LÝ!")

    # =========================
    # Xử lý dữ liệu âm
    # =========================
    # - Với dữ liệu thời tiết, nhiều chỉ số không nên âm (mưa, độ ẩm, tầm nhìn...)
    # - Bạn chọn chiến lược: nếu < 0 thì set về 0
    # - negative_fixed lưu log số lượng đã fix theo từng cột
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
    # - Mục tiêu:
    #   1) Nếu cột dạng string nhưng có thể parse thành số thì chuyển thành numeric
    #   2) Nếu cột có vẻ là thời gian/ngày tháng thì parse thành datetime
    #
    # dtype_log sẽ lưu lại cột nào được chuyển, chuyển theo kiểu gì
    dtype_log = {}
    for col in data_df.columns:
        if data_df[col].dtype == object:
            # Kiểm tra nếu cột có thể chuyển thành số
            try:
                # to_numeric(errors="coerce"):
                # - parse được số thì ra số
                # - parse không được thì thành NaN
                converted = pd.to_numeric(data_df[col], errors="coerce")

                not_null_count = converted.notna().sum()
                total_count = len(data_df[col])
                ratio = not_null_count / total_count if total_count > 0 else 0
                
                # Nếu parse được >= 85% giá trị thì coi như cột này "thực chất là số"
                # -> chuyển sang numeric để phục vụ ML/analytics
                if ratio >= 0.85 and not_null_count > 0:
                    data_df[col] = converted
                    dtype_log[col] = f"string → numeric (parsed {ratio:.0%})"
            except:
                # Nếu lỗi parse -> bỏ qua (giữ nguyên dạng object)
                pass

        # Heuristic nhận diện cột thời gian:
        # - nếu tên cột chứa date/time/thời gian/ngày thì thử parse datetime
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
    # - Nếu data_df rỗng (do drop hết) thì không fill nữa
    if len(data_df) > 0:
        # Xác định nhóm cột:
        # - num_cols: cột số
        # - dt_cols: cột datetime
        # - cat_cols: các cột còn lại (categorical/text)
        num_cols = data_df.select_dtypes(include=[np.number]).columns.tolist()
        dt_cols = data_df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
        cat_cols = [c for c in data_df.columns if c not in num_cols and c not in dt_cols]

        # Fill missing numeric bằng mean của cột
        # - nếu mean NaN (cột toàn NaN) -> fill 0
        for c in num_cols:
            if c in data_df.columns and len(data_df[c]) > 0:
                m = data_df[c].mean()
                data_df[c] = data_df[c].fillna(m if pd.notna(m) else 0)

        # Fill missing datetime bằng mode (giá trị xuất hiện nhiều nhất)
        # - nếu mode rỗng -> dùng pd.NaT
        for c in dt_cols:
            if c in data_df.columns and len(data_df[c]) > 0:
                mode = data_df[c].mode()
                data_df[c] = data_df[c].fillna(mode.iloc[0] if not mode.empty else pd.NaT)

        # Fill missing categorical bằng mode
        # - nếu mode rỗng -> dùng chuỗi rỗng ""
        for c in cat_cols:
            if c in data_df.columns and len(data_df[c]) > 0:
                mode = data_df[c].mode()
                data_df[c] = data_df[c].fillna(mode.iloc[0] if not mode.empty else "")
    else:
        # Nếu không có dữ liệu, trả về DataFrame rỗng
        # - View sẽ trả JSON message cho frontend
        return {
            "message": "Không có dữ liệu sau khi làm sạch",
            "output_file": None,
            "report_file": None
        }

    # =========================
    # So sánh trước – sau
    # =========================
    # - original_missing_like: bản gốc nhưng đã chuẩn hoá missing "giống missing" thành NaN
    # - comparison lưu thống kê để xem cleaning có tác động thế nào
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
    # - Nếu df rỗng thì không xuất file
    if len(data_df) == 0:
        return {
            "message": "Không có dữ liệu để xuất file",
            "output_file": None,
            "report_file": None
        }


    today = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_filename = f"{os.path.splitext(filename)[0]}_cleaned_{today}.csv"

    # Lưu cleaned vào đúng thư mục tuyệt đối như yêu cầu
    # Dynamic path: tự tính từ vị trí project root
    _project_root_dir2 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if file_type == "merged":
        output_dir = os.path.join(_project_root_dir2, "data", "data_clean", "data_merge_clean")
    else:
        output_dir = os.path.join(_project_root_dir2, "data", "data_clean", "data_not_merge_clean")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, clean_filename)

    # Lưu CSV:
    # - index=False: không ghi cột index
    # - utf-8-sig: để Excel mở không bị lỗi tiếng Việt/BOM
    data_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # =========================
    # Xuất báo cáo JSON
    # =========================
    # - report_path: cùng tên với csv nhưng thêm "_report.json"
    # - report gồm:
    #   + comparison: thống kê trước-sau
    #   + cleaning_log: log các bước đã làm (negative_fixed, dtype conversions,...)
    #   + sample_data: 5 dòng đầu để frontend preview nhanh
    report_path = output_path.replace(".csv", "_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "comparison": comparison,
            "cleaning_log": cleaning_log,
            "sample_data": data_df.head(5).to_dict(orient="records") if len(data_df) > 0 else []
        }, f, ensure_ascii=False, indent=4)

    # Return kết quả cho view:
    # - output_file: tên file cleaned để frontend download/hiển thị
    # - report_file: tên file report
    # - rows_remaining: số dòng còn lại sau cleaning
    return {
        "message": "Làm sạch dữ liệu hoàn tất",
        "output_file": clean_filename,
        "report_file": os.path.basename(report_path),
        "rows_remaining": len(data_df)
    }  # (giữ nguyên theo yêu cầu: chỉ thêm chú thích, không đổi logic)
