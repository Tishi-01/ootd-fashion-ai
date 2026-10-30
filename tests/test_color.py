import pytest
import numpy as np
from color_extractor import (
    rgb_to_cielab,
    delta_e,
    match_nearest_shade,
    extract_dominant_colors
)

def test_cielab_conversion():
    white_rgb = np.array([[255, 255, 255]], dtype=np.uint8)
    lab = rgb_to_cielab(white_rgb)
    assert lab.shape == (1, 3)
    assert lab[0, 0] > 95.0 # High luminance

def test_delta_e():
    p1 = (100.0, 0.0, 0.0)
    p2 = (100.0, 0.0, 0.0)
    assert delta_e(p1, p2) == 0.0

    p3 = (50.0, 10.0, 10.0)
    assert delta_e(p1, p3) > 0.0

def test_shade_matching():
    # Pure white vs off white
    white_shade = match_nearest_shade((99.0, 0.0, 0.5))
    assert white_shade["name"] == "pure_white"

    offwhite_shade = match_nearest_shade((93.5, 0.5, 4.0))
    assert offwhite_shade["name"] == "off_white"

    # Denim
    indigo_shade = match_nearest_shade((28.0, 5.0, -28.0))
    assert indigo_shade["name"] == "indigo_denim"

    faded_shade = match_nearest_shade((58.0, 2.0, -22.0))
    assert faded_shade["name"] == "faded_denim"

    # Black
    black_shade = match_nearest_shade((5.0, 0.0, 0.0))
    assert black_shade["name"] == "black"

def test_extract_dominant_colors():
    # Synthetic blue swatch
    blue_img = np.zeros((100, 100, 3), dtype=np.uint8)
    blue_img[:, :] = [30, 60, 140] # RGB blue
    res = extract_dominant_colors(blue_img, is_bgr=False)
    assert "primary" in res
    assert "shade" in res
    assert "palette" in res
    assert len(res["palette"]) >= 1


def test_user_acceptance_color_families():
    # 1. Olive tee -> earthy/green family
    olive_shade = match_nearest_shade((44.0, -2.0, 14.0), subcategory='tshirt')
    assert olive_shade['family'] in ['olive', 'olive_green', 'green', 'earthy']
    assert olive_shade['name'] == 'olive_green'

    # 2. Blue jeans -> cool family, denim/blue shade
    jeans_shade = match_nearest_shade((48.0, -1.0, -8.0), subcategory='jeans')
    assert jeans_shade['family'] in ['blue', 'cool']
    assert jeans_shade['primary'] == 'blue'
    assert jeans_shade['name'] in ['faded_denim', 'indigo_denim', 'blue']

    # 3. White trousers -> neutral family, pure_white / off_white
    trousers_shade = match_nearest_shade((86.0, -0.5, 1.0), subcategory='trousers')
    assert trousers_shade['family'] == 'neutral'
    assert trousers_shade['name'] in ['pure_white', 'off_white']

    # 4. Black top -> neutral family, black
    top_shade = match_nearest_shade((10.0, 0.0, 0.0), subcategory='tank_top')
    assert top_shade['family'] == 'neutral'
    assert top_shade['name'] == 'black'

    # 5. White sandals -> neutral family, pure_white / off_white
    sandals_shade = match_nearest_shade((92.0, 0.0, 2.0), subcategory='sandals')
    assert sandals_shade['family'] == 'neutral'
    assert sandals_shade['name'] in ['pure_white', 'off_white']
