# README Airflow

Tai lieu van hanh Airflow rieng cho project nay.

## Muc tieu
- Tu dong crawl du lieu thoi tiet theo khung gio trong ngay.
- Chi su dung HTML crawler de on dinh.
- File output luu vao `data/data_crawl`.

## Kien truc Docker
Dung chung `docker-compose.yml` voi Django:
- `web` (Django)
- `postgres` (Airflow metadata DB)
- `airflow-init`
- `airflow-webserver`
- `airflow-scheduler`

## DAG va lich crawl
File DAG: `airflow/dags/weather_crawl_schedule.py`

Timezone DAG: `Asia/Bangkok` (UTC+7).

Co 3 DAG:
- `weather_crawl_morning`: `0 5-7 * * *` (05:00, 06:00, 07:00)
- `weather_crawl_noon`: `0,45 12-14 * * *` (12:00, 12:45, 13:00, 13:45, 14:00, 14:45)
- `weather_crawl_evening`: `0 19-21 * * *` (19:00, 20:00, 21:00)

Moi DAG gom 2 task:
1. `crawl_vrain_html`  
   Chay script: `Weather_Forcast_App/scripts/Crawl_data_from_html_of_Vrain.py`
2. `list_latest_outputs`  
   In danh sach file moi nhat trong `data/data_crawl`

## Chuan bi
1. Tao file env:
```powershell
copy .env.example .env
```
2. Dam bao bien admin Airflow ton tai trong `.env`:
- `AIRFLOW_ADMIN_USERNAME`
- `AIRFLOW_ADMIN_PASSWORD`
- `AIRFLOW_ADMIN_EMAIL`

## Khoi dong
```powershell
docker compose up airflow-init
docker compose up -d
```

## Truy cap
- Django: `http://localhost:8000`
- Airflow UI: `http://localhost:8080`

Neu chua doi env:
- user: `admin`
- pass: `admin`

## Kiem tra he thong
Trang thai container:
```powershell
docker compose ps
```

Kiem tra file crawl moi nhat:
```powershell
Get-ChildItem data/data_crawl |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 10 Name,LastWriteTime,Length
```

## Luu y ve thoi gian trong file CSV
Trong file output:
- `timestamp`: thoi diem du lieu mua tu nguon Vrain (vd `19:00`)
- `data_time`: thoi diem script crawl ghi nhan (da doi sang gio VN UTC+7)

## Trigger test thu cong
```powershell
docker exec project_weather_forcast-airflow-scheduler-1 airflow dags trigger weather_crawl_evening
```

## Log thuong dung
- Scheduler DAG parse:
  - `airflow/logs/scheduler/latest/weather_crawl_schedule.py.log`
- Task run:
  - `airflow/logs/dag_id=<dag_id>/run_id=<run_id>/task_id=<task_id>/attempt=*.log`

## Loi thuong gap
- `ModuleNotFoundError: selenium`: image Airflow chua rebuild dependency.
- `NameError: VN_TZ is not defined`: script HTML crawler chua dong bo ban moi.
- DAG khong no ngay gio mong muon: kiem tra timezone va cron trong UI.

## Dung he thong
```powershell
docker compose down
```
