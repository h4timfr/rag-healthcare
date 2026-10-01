"""src/ui/icons.py: the brand marks must be valid, self-contained SVG that
Streamlit accepts as a chat avatar, and their white strokes must stay legible
on their tiles in both themes."""

import xml.etree.ElementTree as ET
from urllib.parse import unquote

import pytest

from src.ui import icons


def _contrast(hex_a: str, hex_b: str) -> float:
    def lum(hex_color: str) -> float:
        channels = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    a, b = sorted((lum(hex_a), lum(hex_b)), reverse=True)
    return (a + 0.05) / (b + 0.05)


@pytest.mark.parametrize("svg", [icons.logo_svg(32), icons.assistant_avatar_svg(), icons.user_avatar_svg()])
def test_brand_marks_are_self_contained_svg(svg):
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert "xmlns" in svg
    assert "http" not in svg.replace("http://www.w3.org/2000/svg", "")  # no external references


@pytest.mark.parametrize("svg", [icons.assistant_avatar_svg(), icons.user_avatar_svg()])
def test_streamlit_accepts_the_marks_as_chat_avatars(svg):
    from streamlit.elements.widgets.chat import _process_avatar_input

    avatar_type, data = _process_avatar_input(svg, "test")
    assert data.startswith("data:image/svg+xml")


@pytest.mark.parametrize("tile", [icons.BRAND_BLUE, icons.USER_SLATE])
def test_white_strokes_keep_aa_contrast_on_their_tile(tile):
    assert _contrast("#ffffff", tile) >= 4.5


def test_logo_data_uri_round_trips():
    uri = icons.svg_data_uri(icons.logo_svg(64))
    assert uri.startswith("data:image/svg+xml;utf8,")
    assert unquote(uri.split(",", 1)[1]) == icons.logo_svg(64)


def test_material_glyph_is_decorative():
    assert icons.material("upload_file") == '<span class="ms" aria-hidden="true">upload_file</span>'


def test_every_icon_value_is_a_material_symbol():
    assert all(v.startswith(":material/") and v.endswith(":") for v in icons.ICON.values())
