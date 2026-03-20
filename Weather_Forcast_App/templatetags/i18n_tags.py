"""
Custom Django template tags for i18n.

Load in templates with:
    {% load i18n_tags %}

Then use:
    {% t "auth.login.title" %}          — outputs translated string
    {% t "nav.home" as label %}{{ label }}  — assigns to variable
    {{ t("nav.home") }}                 — works via context processor
"""

from django import template
from django.utils.safestring import mark_safe

from Weather_Forcast_App.i18n import translate, detect_language

register = template.Library()


class TranslateNode(template.Node):
    def __init__(self, key_expr, var_name=None):
        self.key_expr = key_expr
        self.var_name = var_name

    def render(self, context):
        key = self.key_expr.resolve(context)
        # Prefer lang from context (set by context processor / middleware)
        lang = context.get("lang") or "vi"
        result = translate(str(key), lang)
        if self.var_name:
            context[self.var_name] = result
            return ""
        return mark_safe(result)


@register.tag("t")
def do_translate(parser, token):
    """
    Translate a key.

    Usage:
        {% t "some.key" %}
        {% t "some.key" as varname %}
        {% t key_variable %}
    """
    bits = token.split_contents()
    tag_name = bits[0]

    if len(bits) < 2:
        raise template.TemplateSyntaxError(
            f"'{tag_name}' tag requires at least one argument (translation key)"
        )

    key_expr = parser.compile_filter(bits[1])
    var_name = None

    if len(bits) == 4 and bits[2] == "as":
        var_name = bits[3]
    elif len(bits) != 2:
        raise template.TemplateSyntaxError(
            f"'{tag_name}' tag syntax: {{% t 'key' %}} or {{% t 'key' as varname %}}"
        )

    return TranslateNode(key_expr, var_name)


@register.simple_tag(takes_context=True)
def lang_url(context, lang_code):
    """
    Return the current URL with ?lang=<lang_code> appended.

    Usage:
        <a href="{% lang_url 'en' %}">EN</a>
    """
    request = context.get("request")
    if request is None:
        return f"?lang={lang_code}"

    # Preserve existing query params except 'lang'
    params = request.GET.copy()
    params["lang"] = lang_code
    return f"{request.path}?{params.urlencode()}"