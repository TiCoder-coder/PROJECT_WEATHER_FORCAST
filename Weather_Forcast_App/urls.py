from django.urls import path
from .views.Home import home_view
from .views.View_Crawl_data_by_API import crawl_api_weather_view, api_weather_logs_view
from .views.View_Crawl_data_by_API import crawl_vrain_html_view
from .views.View_Datasets import datasets_view, dataset_download_view, dataset_view_view
app_name = "weather"

urlpatterns = [
    path("", home_view, name="home"),

    path("crawl-api-weather/", crawl_api_weather_view, name="crawl_api_weather"),
    path("crawl-by-api/", crawl_api_weather_view, name="crawl_by_api"),

    path("crawl-api-weather/logs/", api_weather_logs_view, name="crawl_api_weather_logs"),

    path("crawl-vrain-html/", crawl_vrain_html_view, name="crawl_vrain_html"),
    path("datasets/", datasets_view, name="datasets"),
    path("datasets/view/<str:filename>/", dataset_view_view, name="dataset_view"),
    path("datasets/download/<str:filename>/", dataset_download_view, name="dataset_download"),
]
