from django.urls import path
from .views.Home import home_view
from .views.View_Crawl_data_by_API import crawl_api_weather_view, api_weather_logs_view
from .views.View_Datasets import datasets_view, dataset_download_view, dataset_view_view
from .views.View_Merge_Data import merge_data_view
from .views.View_Clear import clean_data_view
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
from .views.View_Clear import (
    clean_data_view,
    clean_files_list_view,
    clean_data_tail_view,
)

from .views.View_login import (
    login_view, register_view, logout_view, profile_view,
    forgot_password_view, password_reset_sent_view,
    reset_password_view, password_reset_complete_view,
    forgot_password_otp_view, verify_otp_view, reset_password_otp_view,
    verify_email_register_view, resend_email_otp_view, cancel_register_view,
)

app_name = "weather"

urlpatterns = [
    path("", home_view, name="home"),
    
    path("crawl-api-weather/", crawl_api_weather_view, name="crawl_api_weather"),
    path("crawl-by-api/", crawl_api_weather_view, name="crawl_by_api"),
    
    path("crawl-api-weather/logs/", api_weather_logs_view, name="crawl_api_weather_logs"),
    
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
    path("datasets/view/<str:folder>/<str:filename>/", dataset_view_view, name="dataset_view"),
    path("datasets/download/<str:folder>/<str:filename>/", dataset_download_view, name="dataset_download"),
    
    path("datasets/merge/", merge_data_view, name="merge_data"),
    
    path("datasets/clean/", clean_data_view, name="clean_data"),
    path("datasets/clean/list/", clean_files_list_view, name="clean_list"),
    path("datasets/clean/tail/", clean_data_tail_view, name="clean_tail"),
    
    path("auth/login/", login_view, name="login"),
    path("auth/register/", register_view, name="register"),
    path("auth/logout/", logout_view, name="logout"),
    path("auth/profile/", profile_view, name="profile"),
    path("auth/forgot-password/", forgot_password_view, name="forgot_password"),
    path("auth/password-reset-sent/", password_reset_sent_view, name="password_reset_sent"),

    # OTP flow
    path("auth/forgot-password-otp/", forgot_password_otp_view, name="forgot_password_otp"),
    path("auth/verify-otp/", verify_otp_view, name="verify_otp"),
    path("auth/reset-password-otp/", reset_password_otp_view, name="reset_password_otp"),

    # Email verification for registration
    path("auth/verify-email-register/", verify_email_register_view, name="verify_email_register"),
    path("auth/resend-email-otp/", resend_email_otp_view, name="resend_email_otp"),
    path("auth/cancel-register/", cancel_register_view, name="cancel_register"),

]
