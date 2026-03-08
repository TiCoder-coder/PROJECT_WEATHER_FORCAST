"""
Language-detection middleware.

Sets ``request.lang`` and ``request.t`` on every request so that
views can call ``request.t("some.key")`` without importing i18n directly.
"""

from Weather_Forcast_App.i18n import detect_language, translate, SUPPORTED_LANGS


class LangMiddleware:
    """
    Lightweight middleware that attaches language helpers to the request.

    Registered in settings.py MIDDLEWARE list *after* SessionMiddleware so
    that session-based language preferences are available.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        lang = detect_language(request)
        request.lang = lang

        def _t(key: str) -> str:
            return translate(key, lang)

        _t.lang = lang
        request.t = _t

        response = self.get_response(request)
        return response
