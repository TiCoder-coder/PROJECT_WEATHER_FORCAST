from __future__ import annotations

import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.timetables.trigger import CronTriggerTimetable


LOCAL_TZ = pendulum.timezone("Asia/Bangkok")

default_args = {"owner": "weather-team", "depends_on_past": False, "retries": 1}


with DAG(
    dag_id="weather_api_crawl_hourly",
    default_args=default_args,
    description="Run Crawl_data_by_API.py every hour",
    start_date=pendulum.datetime(2026, 3, 9, tz=LOCAL_TZ),
    schedule=CronTriggerTimetable("0 * * * *", timezone=LOCAL_TZ),
    catchup=False,
    max_active_runs=1,
    tags=["weather", "crawl", "api"],
) as dag:
    crawl_weather_by_api = BashOperator(
        task_id="crawl_weather_by_api",
        bash_command=(
            "cd /opt/project && "
            "CRAWL_MODE=once python Weather_Forcast_App/scripts/Crawl_data_by_API.py"
        ),
    )

    trigger_dedupe = TriggerDagRunOperator(
        task_id="trigger_dedupe_crawl",
        trigger_dag_id="weather_dedupe_crawl",
        wait_for_completion=False,
    )

    crawl_weather_by_api >> trigger_dedupe
