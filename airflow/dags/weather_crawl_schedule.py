from __future__ import annotations

import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.timetables.trigger import CronTriggerTimetable


LOCAL_TZ = pendulum.timezone("Asia/Bangkok")

default_args = {"owner": "weather-team", "depends_on_past": False, "retries": 1}


def build_crawl_dag(dag_id: str, schedule: str, description: str) -> DAG:
    with DAG(
        dag_id=dag_id,
        default_args=default_args,
        description=description,
        start_date=pendulum.datetime(2026, 3, 8, tz=LOCAL_TZ),
        schedule=CronTriggerTimetable(schedule, timezone=LOCAL_TZ),
        catchup=False,
        max_active_runs=1,
        tags=["weather", "crawl"],
    ) as dag:
        # C1: Crawl VRAIN bằng HTML Parser (Selenium + Regex)
        crawl_vrain_html = BashOperator(
            task_id="crawl_vrain_html",
            bash_command=(
                "cd /opt/project && "
                "python Weather_Forcast_App/scripts/Crawl_data_from_html_of_Vrain.py"
            ),
        )

        # C2: Crawl VRAIN bằng Selenium (parse element trực tiếp)
        crawl_vrain_selenium = BashOperator(
            task_id="crawl_vrain_selenium",
            bash_command=(
                "cd /opt/project && "
                "python Weather_Forcast_App/scripts/Crawl_data_from_Vrain_by_Selenium.py"
            ),
        )

        # C3: Crawl VRAIN bằng REST API
        crawl_vrain_api = BashOperator(
            task_id="crawl_vrain_api",
            bash_command=(
                "cd /opt/project && "
                "python Weather_Forcast_App/scripts/Crawl_data_from_Vrain_by_API.py"
            ),
        )

        # C4: Crawl OpenWeather / WeatherAPI (chạy 1 lần, không loop)
        crawl_openweather_api = BashOperator(
            task_id="crawl_openweather_api",
            bash_command=(
                "cd /opt/project && "
                "python Weather_Forcast_App/scripts/Crawl_data_by_API.py --mode once"
            ),
            env={"CRAWL_MODE": "once"},
        )

        list_latest_outputs = BashOperator(
            task_id="list_latest_outputs",
            bash_command=(
                "cd /opt/project && "
                "python -c \"from pathlib import Path; "
                "p=Path('data/data_crawl'); "
                "files=sorted([f for f in p.glob('Bao_cao_*') if f.is_file()], "
                "key=lambda x: x.stat().st_mtime, reverse=True)[:10]; "
                "print('\\n'.join(str(f) for f in files) if files else 'No output files found')\""
            ),
        )

        # Tất cả 4 crawl chạy song song, xong mới tổng kết output
        [crawl_vrain_html, crawl_vrain_selenium, crawl_vrain_api, crawl_openweather_api] >> list_latest_outputs
        return dag


# Sang som: 05h-07h, moi 1 gio
weather_crawl_morning = build_crawl_dag(
    dag_id="weather_crawl_morning",
    schedule="0 5-7 * * *",
    description="Morning crawl window: every 1 hour from 05:00 to 07:00",
)

# Buoi trua: 12h-14h, moi 45 phut
weather_crawl_noon = build_crawl_dag(
    dag_id="weather_crawl_noon",
    schedule="0,45 12-14 * * *",
    description="Noon crawl window: every 45 minutes from 12:00 to 14:59",
)

# Buoi toi: 19h-21h, lay tu 19h tro di moi 1 gio
weather_crawl_evening = build_crawl_dag(
    dag_id="weather_crawl_evening",
    schedule="0 19-21 * * *",
    description="Evening crawl window: every 1 hour from 19:00 to 21:00",
)
