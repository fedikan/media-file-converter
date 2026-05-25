"""Localized strings drawn directly onto OG cards by compose.py.

Kept tiny on purpose — these are template-level labels (chip text,
"by", "untitled" placeholders, node-count plurals). User-facing copy
that flows from the frontend (canvas name, prompt, model description)
already arrives in the right language; this module only fills in the
gaps the renderer adds itself.

Usage:
    from .i18n import T, plural
    T("canvas.chip", locale)              # "CANVAS" / "ХОЛСТ"
    T("model.by", locale, author="Alice") # "by Alice" / "от Alice"
    plural(n, "canvas.nodes", locale)     # "5 nodes" / "5 узлов"
"""

# Default locale used when an unknown / missing locale is requested.
DEFAULT_LOCALE = "en"
SUPPORTED_LOCALES = ("en", "ru")


# --- Singular strings (template substitution via str.format if **fmt given) ---
STRINGS = {
    # Canvas card
    "canvas.chip": {
        "en": "CANVAS",
        "ru": "ХОЛСТ",
    },
    "canvas.untitledName": {
        "en": "Untitled canvas",
        "ru": "Холст без названия",
    },

    # Model card
    "model.fallbackName": {
        "en": "Model",
        "ru": "Модель",
    },
    "model.by": {
        "en": "by {author}",
        "ru": "от {author}",
    },
    # Model type chips. The frontend passes the raw type string (e.g.
    # "image"); we render an uppercased localized label. Unknown types
    # fall through to the original string uppercased.
    "model.type.image": {
        "en": "IMAGE",
        "ru": "ИЗОБРАЖЕНИЕ",
    },
    "model.type.video": {
        "en": "VIDEO",
        "ru": "ВИДЕО",
    },
    "model.type.audio": {
        "en": "AUDIO",
        "ru": "АУДИО",
    },
    "model.type.text": {
        "en": "TEXT",
        "ru": "ТЕКСТ",
    },
    "model.type.3d": {
        "en": "3D",
        "ru": "3D",
    },

    # Generation card
    "generation.agentLabel": {
        "en": "AGENT",
        "ru": "АГЕНТ",
    },
    "generation.agentRunFallback": {
        "en": "AGENT RUN",
        "ru": "АГЕНТ-СЕССИЯ",
    },
    "generation.brandFallback": {
        "en": "Ropewalk",
        "ru": "Ropewalk",
    },
}


# --- Plural forms.
# English: { one, other }. Russian (CLDR): { one, few, many }.
# We use the simplified Russian rule that's correct for non-fractional counts.
PLURALS = {
    "canvas.nodes": {
        "en": {"one": "{n} node", "other": "{n} nodes"},
        "ru": {
            "one":  "{n} узел",   # 1, 21, 31...
            "few":  "{n} узла",   # 2-4, 22-24...
            "many": "{n} узлов",  # 5-20, 25-30...
        },
    },
}


def _pick_locale(locale):
    if locale in SUPPORTED_LOCALES:
        return locale
    return DEFAULT_LOCALE


def T(key: str, locale: str = DEFAULT_LOCALE, **fmt) -> str:
    """Lookup a localized string and apply str.format substitutions."""
    loc = _pick_locale(locale)
    entry = STRINGS.get(key)
    if not entry:
        # Unknown key: surface the key itself so it's grep-able in renders.
        return key
    template = entry.get(loc) or entry.get(DEFAULT_LOCALE) or key
    if fmt:
        try:
            return template.format(**fmt)
        except (KeyError, IndexError):
            return template
    return template


def _ru_plural_form(n: int) -> str:
    """Russian plural category for an integer count."""
    n = abs(int(n))
    mod10 = n % 10
    mod100 = n % 100
    if mod10 == 1 and mod100 != 11:
        return "one"
    if mod10 in (2, 3, 4) and mod100 not in (12, 13, 14):
        return "few"
    return "many"


def _en_plural_form(n: int) -> str:
    return "one" if abs(int(n)) == 1 else "other"


def plural(n: int, key: str, locale: str = DEFAULT_LOCALE) -> str:
    """Return a count-aware localized phrase, e.g. plural(5, 'canvas.nodes', 'ru') → '5 узлов'."""
    loc = _pick_locale(locale)
    entry = PLURALS.get(key)
    if not entry:
        return f"{n}"
    table = entry.get(loc) or entry.get(DEFAULT_LOCALE) or {}
    if loc == "ru":
        form = _ru_plural_form(n)
    else:
        form = _en_plural_form(n)
    template = table.get(form) or table.get("other") or table.get("one") or "{n}"
    try:
        return template.format(n=n)
    except (KeyError, IndexError):
        return f"{n}"


def model_type_label(model_type: str, locale: str = DEFAULT_LOCALE) -> str:
    """Localize a model-type string ("image", "Video", etc.) for the chip.
    Falls back to the original uppercased if the type is unknown."""
    if not model_type:
        return ""
    key = f"model.type.{model_type.strip().lower()}"
    entry = STRINGS.get(key)
    if entry:
        return T(key, locale)
    return model_type.strip().upper()
