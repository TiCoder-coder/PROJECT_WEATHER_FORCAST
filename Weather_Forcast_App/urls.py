from django.urls import path
from .views.Home import home_view
from .views.View_Crawl_data_by_API import crawl_api_weather_view, api_weather_logs_view
from .views.View_Crawl_data_by_API import crawl_vrain_html_view
from .views.View_Datasets import datasets_view, dataset_download_view, dataset_view_view
from .views.View_Merge_Data import merge_data_view
from .views.View_Crawl_data_from_html_of_Vrain import (
    crawl_vrain_html_view,
    crawl_vrain_html_start_view,
    crawl_vrain_html_tail_view,
)
from .views.View_Crawl_data_from_html_of_Vrain import (
    crawl_vrain_html_view,
    crawl_vrain_html_start_view,
    crawl_vrain_html_tail_view,
)
from .views.View_Crawl_data_from_Vrain_by_API import (
    crawl_vrain_api_view,
    crawl_vrain_api_start_view,
    crawl_vrain_api_tail_view,
)
from .views.View_Crawl_data_from_Vrain_by_Selenium import (
    crawl_vrain_selenium_view,
    crawl_vrain_selenium_start_view,
    crawl_vrain_selenium_tail_view,
)
app_name = "weather"

urlpatterns = [
    path("", home_view, name="home"),

    path("crawl-api-weather/", crawl_api_weather_view, name="crawl_api_weather"),
    path("crawl-by-api/", crawl_api_weather_view, name="crawl_by_api"),

    path("crawl-api-weather/logs/", api_weather_logs_view, name="crawl_api_weather_logs"),

    path("crawl-vrain-html/", crawl_vrain_html_view, name="crawl_vrain_html"),


    path("crawl-vrain-html/", crawl_vrain_html_view, name="crawl_vrain_html"),
    path("crawl-vrain-html/start/", crawl_vrain_html_start_view, name="crawl_vrain_html_start"),
    path("crawl-vrain-html/tail/", crawl_vrain_html_tail_view, name="crawl_vrain_html_tail"),

    path("crawl-vrain-html/", crawl_vrain_html_view, name="crawl_vrain_html"),
    path("crawl-vrain-html/start/", crawl_vrain_html_start_view, name="crawl_vrain_html_start"),
    path("crawl-vrain-html/tail/", crawl_vrain_html_tail_view, name="crawl_vrain_html_tail"),
    
    path("crawl-vrain-api/", crawl_vrain_api_view, name="crawl_vrain_api"),
    path("crawl-vrain-api/start/", crawl_vrain_api_start_view, name="crawl_vrain_api_start"),
    path("crawl-vrain-api/tail/", crawl_vrain_api_tail_view, name="crawl_vrain_api_tail"),

    path("crawl-vrain-selenium/", crawl_vrain_selenium_view, name="crawl_vrain_selenium"),
    path("crawl-vrain-selenium/start/", crawl_vrain_selenium_start_view, name="crawl_vrain_selenium_start"),
    path("crawl-vrain-selenium/tail/", crawl_vrain_selenium_tail_view, name="crawl_vrain_selenium_tail"),



    path("datasets/", datasets_view, name="datasets"),
    path("datasets/view/<str:filename>/", dataset_view_view, name="dataset_view"),
    path("datasets/download/<str:filename>/", dataset_download_view, name="dataset_download"),
    
    # Merge data endpoint
    path("datasets/merge/", merge_data_view, name="merge_data"),
]