"""The app's icon language, in one place.

Two kinds of icon, deliberately:

- Brand marks (the logo, and the two chat avatars) are small inline SVGs
  drawn on a 32px grid with 1.8px rounded strokes. They are rendered as
  images (chat avatars, the title mark), so their colors are fixed rather
  than theme variables - chosen to read on both the dark and light themes.

- Every UI control icon (upload, remove, retry, sources, evidence, ...) is a
  Material Symbols Rounded glyph: the icon font Streamlit already ships and
  uses for its own controls, so buttons, expanders and the custom HTML below
  all share one family. Nothing is fetched from a CDN.
"""

from urllib.parse import quote

# Brand blue: the app's accent, one step darker so white strokes on it
# keep >= 4.5:1 contrast. Neutral slate for the user's avatar.
BRAND_BLUE = "#1f6feb"
USER_SLATE = "#57606a"

_STROKE = 'fill="none" stroke="#ffffff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"'

# A page with a folded corner, one line of text, and a pulse line that ends
# in a node: a document, a clinical signal in it, and the point an answer is
# grounded on.
_LOGO_GLYPH = (
    f'<path d="M10.5 6.5h8.25l4.75 4.75V24a1.5 1.5 0 0 1-1.5 1.5h-11.5A1.5 1.5 0 0 1 9 24V8'
    f'a1.5 1.5 0 0 1 1.5-1.5z" {_STROKE}/>'
    f'<path d="M18.5 6.75v4.75h4.75" {_STROKE}/>'
    f'<path d="M12 12.75h4" {_STROKE}/>'
    f'<path d="M11.75 19h2l1.5-3.25 2.25 6 1.35-2.75" {_STROKE}/>'
    '<circle cx="21" cy="19" r="1.3" fill="#ffffff"/>'
)

# Head and shoulders, same grid and stroke as the logo glyph.
_USER_GLYPH = (
    f'<circle cx="16" cy="12.5" r="4.25" {_STROKE}/>'
    f'<path d="M8.75 24.5c.9-3.9 3.75-6 7.25-6s6.35 2.1 7.25 6" {_STROKE}/>'
)


def _tile(glyph: str, fill: str, size: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="{size}" height="{size}">'
        f'<rect width="32" height="32" rx="8" fill="{fill}"/>{glyph}</svg>'
    )


def logo_svg(size: int = 32) -> str:
    return _tile(_LOGO_GLYPH, BRAND_BLUE, size)


def assistant_avatar_svg() -> str:
    """The answer's avatar is the logo mark: an answer is the project's
    output, drawn from the retrieved document evidence."""
    return logo_svg(32)


def user_avatar_svg() -> str:
    return _tile(_USER_GLYPH, USER_SLATE, 32)


def svg_data_uri(svg: str) -> str:
    """For CSS `url(...)`; percent-encoded rather than base64 so it stays readable."""
    return "data:image/svg+xml;utf8," + quote(svg, safe="/:=,.-' ")


def material(name: str, extra_class: str = "") -> str:
    """An inline Material Symbols Rounded glyph for custom HTML, decorative
    (aria-hidden) - the text next to it carries the meaning."""
    cls = f"ms {extra_class}".strip()
    return f'<span class="{cls}" aria-hidden="true">{name}</span>'


# Streamlit `icon=` values for buttons and expanders, so each action uses the
# same glyph everywhere it appears.
ICON = {
    "remove": ":material/delete:",
    "remove_confirm": ":material/delete_forever:",
    "cancel": ":material/close:",
    "dismiss": ":material/close:",
    "retry": ":material/refresh:",
    "selected": ":material/check_circle:",
    "clear": ":material/delete_sweep:",
    "evidence": ":material/format_quote:",
    "timing": ":material/schedule:",
    "details": ":material/code:",
    "about": ":material/info:",
    "removed": ":material/check_circle:",
    "failed": ":material/error:",
}
