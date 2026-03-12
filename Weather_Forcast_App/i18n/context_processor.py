"""
i18n context processor.

Injects ``t`` (translation function) and ``lang`` (current language code)
into every template context, so templates can call {{ t("key") }} directly
without any extra setup.

Registration in settings.py:
    TEMPLATES[0]["OPTIONS"]["context_processors"] += [
        "Weather_Forcast_App.i18n.context_processor.i18n_context",
    ]
"""

from Weather_Forcast_App.i18n import get_t, SUPPORTED_LANGS


def i18n_context(request):
    """
    Returns context variables available in every template:

    - ``t``         callable(key) → translated string
    - ``lang``      current language code, e.g. "vi" or "en"
    - ``languages`` list of supported language codes
    """
    t = get_t(request)
    return {
        "t": t,
        "lang": t.lang,
        "languages": SUPPORTED_LANGS,
    }
