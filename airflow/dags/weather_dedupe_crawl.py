from __future__ import annotations

import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.timetables.trigger import CronTriggerTimetable


LOCAL_TZ = pendulum.timezone("Asia/Bangkok")

default_args = {"owner": "weather-team", "depends_on_past": False, "retries": 1}


with DAG(
    dag_id="weather_dedupe_crawl",
    default_args=default_args,
    description="Deduplicate latest crawl data and write to data_clean",
    start_date=pendulum.datetime(2026, 3, 11, tz=LOCAL_TZ),
    schedule=CronTriggerTimetable("10 * * * *", timezone=LOCAL_TZ),
    catchup=False,
    max_active_runs=1,
    tags=["weather", "clean", "dedupe"],
) as dag:
    dedupe_latest_crawl = BashOperator(
        task_id="dedupe_latest_crawl",
        bash_command=(
            "cd /opt/project && "
            "python Weather_Forcast_App/scripts/dedupe_latest_crawl.py"
        ),
    )
