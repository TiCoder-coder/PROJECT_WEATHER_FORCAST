"""
i18n — Python multilingual support module for Weather_Forcast_App.

Mirrors the TypeScript i18next pattern (index.ts / hooks.ts) but in pure Python.

Usage in views:
    from Weather_Forcast_App.i18n import get_t
    t = get_t(request)
    label = t("auth.login.title")

Usage in templates (via context processor):
    {{ t("auth.login.title") }}
    {% t "auth.login.title" %}   (requires {% load i18n_tags %})

Language resolution order:
    1. request.session["lang"]
    2. ?lang= query param  (also stored into session)
    3. Accept-Language header  (first code, e.g. "vi" or "en")
    4. fallback → "vi"
"""

import json
import os
from typing import Any

SUPPORTED_LANGS = ("vi", "en")
FALLBACK_LANG = "vi"

_LOCALES_DIR = os.path.join(os.path.dirname(__file__), "..", "locales")

# Cache: lang -> (mtime, data)
_locale_cache: dict[str, tuple[float, dict]] = {}


def _load_locale(lang: str) -> dict:
    """Load locale JSON file, reloading automatically if the file has changed."""
    path = os.path.join(_LOCALES_DIR, f"{lang}.json")
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return {}

    cached = _locale_cache.get(lang)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    _locale_cache[lang] = (mtime, data)
    return data


def _get(data: Any, keys: list[str]) -> Any:
    """Traverse nested dict by key list."""
    for key in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def translate(key: str, lang: str) -> str:
    """
    Look up *key* (dot-separated path) in the locale for *lang*.
    Falls back to FALLBACK_LANG, then returns the key itself.
    """
    if lang not in SUPPORTED_LANGS:
        lang = FALLBACK_LANG

    parts = key.split(".")
    value = _get(_load_locale(lang), parts)

    if value is None and lang != FALLBACK_LANG:
        value = _get(_load_locale(FALLBACK_LANG), parts)

    if value is None:
        return key  # last-resort: return key unchanged

    return str(value)


def detect_language(request) -> str:
    """
    Detect the preferred language from the request.

    Priority:
        ?lang= query param  →  session  →  Accept-Language  →  FALLBACK_LANG
    """
    # 1. URL query param — also stores choice in session
    param = request.GET.get("lang")
    if param in SUPPORTED_LANGS:
        request.session["lang"] = param
        return param

    # 2. Session
    session_lang = request.session.get("lang")
    if session_lang in SUPPORTED_LANGS:
        return session_lang

    # 3. Accept-Language header  (e.g. "en-US,en;q=0.9,vi;q=0.8")
    accept = request.META.get("HTTP_ACCEPT_LANGUAGE", "")
    for token in accept.replace("-", "_").split(","):
        code = token.strip().split(";")[0].split("_")[0].lower()
        if code in SUPPORTED_LANGS:
            return code

    return FALLBACK_LANG


def get_t(request):
    """
    Return a bound translation function for *request*.

    The returned callable accepts a dot-path key and returns the
    translated string in the request's detected language.

    Example:
        t = get_t(request)
        t("auth.login.title")   # → "Đăng nhập"  (if lang == "vi")
    """
    lang = detect_language(request)

    def _t(key: str) -> str:
        return translate(key, lang)

    _t.lang = lang  # expose current lang so templates can read it
    return _t
