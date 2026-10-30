import os
import pytest
from pathlib import Path
from pipeline import WardrobePipeline
from color_extractor import match_nearest_shade

@pytest.fixture(scope="module")
def pipeline():
    return WardrobePipeline.get_instance()

def test_color_unit_tests():
    # 1. Olive tee -> olive/green family name
    olive = match_nearest_shade((44.0, -2.0, 14.0), subcategory="tshirt")
    assert olive["family"] in ["olive_green", "green", "olive", "earthy"]
    assert olive["name"] == "olive_green"

    # 2. Blue jeans -> blue family name
    jeans = match_nearest_shade((48.0, -1.0, -8.0), subcategory="jeans")
    assert jeans["family"] in ["blue", "cool"]
    assert jeans["name"] in ["faded_denim", "indigo_denim", "dark_navy", "blue"]

    # 3. White trousers -> neutral / white family
    trousers = match_nearest_shade((86.0, -0.5, 1.0), subcategory="wide_leg_trousers")
    assert trousers["family"] in ["neutral", "white"]
    assert trousers["name"] in ["pure_white", "off_white"]

    # 4. Black top -> neutral / black family
    top = match_nearest_shade((10.0, 0.0, 0.0), subcategory="tank_top")
    assert top["family"] in ["neutral", "black"]
    assert top["name"] == "black"

    # 5. White sandals -> neutral / white family
    sandals = match_nearest_shade((92.0, 0.0, 2.0), subcategory="platform_sandals")
    assert sandals["family"] in ["neutral", "white"]
    assert sandals["name"] in ["pure_white", "off_white"]

def test_img1_acceptance(pipeline):
    img_path = Path("tests/images/img1.jpg")
    res = pipeline.analyze_outfit(img_path)
    items = res["detected_items"]

    print("\n\n=== Acceptance Test Output: img1.jpg (Engine: " + res["engine"] + ") ===")
    print(f"| {"Item":<24} | {"Subcategory":<18} | {"Color":<12} | {"Pattern":<10} | {"Conf":<6} | {"Engine":<8} |")
    print("|" + "-"*26 + "|" + "-"*20 + "|" + "-"*14 + "|" + "-"*12 + "|" + "-"*8 + "|" + "-"*10 + "|")
    for it in items:
        print(f"| {it['name']:<24} | {it['subcategory']:<18} | {it['shade']:<12} | {it['pattern']:<10} | {it['confidence']:<6.0%} | {it['engine']:<8} |")

    subcats = {it["subcategory"]: it for it in items}

    # 1. Blazer (plaid/checkered, outer layer)
    assert "blazer" in subcats, "img1 must detect blazer"
    blazer = subcats["blazer"]
    assert blazer["layer"] == "outer"
    assert blazer["pattern"] in ["checkered", "plaid"]
    assert blazer["shade"] in ["black", "charcoal_grey", "grey"]

    # 2. Top (black sleeveless, NOT checkered)
    assert any(sub in subcats for sub in ["tank_top", "crop_top", "top"]), "img1 must detect top"
    top_key = next(sub for sub in ["tank_top", "crop_top", "top"] if sub in subcats)
    top = subcats[top_key]
    assert top["shade"] in ["black", "charcoal_grey"]
    assert top["pattern"] != "checkered", "Top must NOT be tagged checkered"

    # 3. Wide-leg trousers (white)
    assert any(sub in subcats for sub in ["wide_leg_trousers", "trousers"]), "img1 must detect wide-leg trousers"
    trousers_key = next(sub for sub in ["wide_leg_trousers", "trousers"] if sub in subcats)
    trousers = subcats[trousers_key]
    assert trousers["shade"] in ["pure_white", "off_white"]

    # 4. Sandals (white, platform)
    assert any(sub in subcats for sub in ["platform_sandals", "sandals"]), "img1 must detect platform sandals"
    sandals_key = next(sub for sub in ["platform_sandals", "sandals"] if sub in subcats)
    sandals = subcats[sandals_key]
    assert sandals["shade"] in ["pure_white", "off_white"]

    # 5. Handbag (black)
    assert "handbag" in subcats, "img1 must detect handbag"
    assert subcats["handbag"]["shade"] == "black"

    # Verify boxes and mask centers
    for it in items:
        ymin, xmin, ymax, xmax = it["bounding_box"]
        cy, cx = it["mask_center"]
        assert ymin <= cy <= ymax, f"{it['subcategory']} mask center y {cy} not in [{ymin}..{ymax}]"
        assert xmin <= cx <= xmax, f"{it['subcategory']} mask center x {cx} not in [{xmin}..{xmax}]"
        area_pct = (ymax - ymin) * (xmax - xmin) / (1024 * 1024)
        assert area_pct <= 0.70, f"{it['subcategory']} box area too large: {area_pct:.1%}"

def test_img2_acceptance(pipeline):
    img_path = Path("tests/images/img2.jpg")
    res = pipeline.analyze_outfit(img_path)
    items = res["detected_items"]

    print("\n\n=== Acceptance Test Output: img2.jpg (Engine: " + res["engine"] + ") ===")
    print(f"| {"Item":<24} | {"Subcategory":<18} | {"Color":<12} | {"Pattern":<10} | {"Conf":<6} | {"Engine":<8} |")
    print("|" + "-"*26 + "|" + "-"*20 + "|" + "-"*14 + "|" + "-"*12 + "|" + "-"*8 + "|" + "-"*10 + "|")
    for it in items:
        print(f"| {it['name']:<24} | {it['subcategory']:<18} | {it['shade']:<12} | {it['pattern']:<10} | {it['confidence']:<6.0%} | {it['engine']:<8} |")

    subcats = {it["subcategory"]: it for it in items}

    # 1. Bomber jacket (white, outer)
    assert any(sub in subcats for sub in ["bomber_jacket", "jacket"]), "img2 must detect bomber jacket"
    jacket_key = next(sub for sub in ["bomber_jacket", "jacket"] if sub in subcats)
    jacket = subcats[jacket_key]
    assert jacket["layer"] == "outer"
    assert jacket["shade"] in ["pure_white", "off_white"]

    # 2. T-shirt (white, inner)
    assert "tshirt" in subcats, "img2 must detect inner t-shirt"
    tshirt = subcats["tshirt"]
    assert tshirt["layer"] == "inner"
    assert tshirt["shade"] in ["pure_white", "off_white"]

    # 3. Trousers or joggers (dark navy)
    assert any(sub in subcats for sub in ["joggers", "trousers", "sweatpants"]), "img2 must detect joggers/trousers"
    pants_key = next(sub for sub in ["joggers", "trousers", "sweatpants"] if sub in subcats)
    pants = subcats[pants_key]
    assert pants["shade"] == "dark_navy" or pants["color"]["family"] == "blue"

    # 4. No phantom items (total items exactly 3)
    assert len(items) == 3, f"img2 has stray phantom items: {[it['subcategory'] for it in items]}"

    # Verify boxes
    h, w = 1100, 1319
    for it in items:
        ymin, xmin, ymax, xmax = it["bounding_box"]
        cy, cx = it["mask_center"]
        assert ymin <= cy <= ymax
        assert xmin <= cx <= xmax
        area_pct = (ymax - ymin) * (xmax - xmin) / (h * w)
        assert area_pct <= 0.70

def test_img3_acceptance(pipeline):
    img_path = Path("tests/images/img3.jpg")
    res = pipeline.analyze_outfit(img_path)
    items = res["detected_items"]

    print("\n\n=== Acceptance Test Output: img3.jpg (Engine: " + res["engine"] + ") ===")
    print(f"| {"Item":<24} | {"Subcategory":<18} | {"Color":<12} | {"Pattern":<10} | {"Conf":<6} | {"Engine":<8} |")
    print("|" + "-"*26 + "|" + "-"*20 + "|" + "-"*14 + "|" + "-"*12 + "|" + "-"*8 + "|" + "-"*10 + "|")
    for it in items:
        print(f"| {it['name']:<24} | {it['subcategory']:<18} | {it['shade']:<12} | {it['pattern']:<10} | {it['confidence']:<6.0%} | {it['engine']:<8} |")

    subcats = {it["subcategory"]: it for it in items}

    # 1. T-shirt (olive green, graphic/printed)
    assert "tshirt" in subcats, "img3 must detect tshirt"
    tshirt = subcats["tshirt"]
    assert tshirt["shade"] == "olive_green"
    assert tshirt["color"]["family"] in ["olive_green", "green", "olive", "earthy"]
    assert tshirt["pattern"] == "graphic"

    # 2. Jeans (blue, slim, denim)
    assert "jeans" in subcats, "img3 must detect jeans"
    jeans = subcats["jeans"]
    assert jeans["material"] == "denim"
    assert jeans["fit_type"] == "slim"
    assert jeans["color"]["family"] in ["blue", "cool"]

    # 3. Sneakers
    assert "sneakers" in subcats, "img3 must detect sneakers"

    # 4. No phantom items (total items exactly 3, no phantom jacket on stool)
    assert "jacket" not in subcats, "img3 must not have phantom jacket on stool"
    assert len(items) == 3, f"img3 has phantom items: {[it['subcategory'] for it in items]}"

    # Verify boxes
    h, w = 1321, 957
    for it in items:
        ymin, xmin, ymax, xmax = it["bounding_box"]
        cy, cx = it["mask_center"]
        assert ymin <= cy <= ymax
        assert xmin <= cx <= xmax
        area_pct = (ymax - ymin) * (xmax - xmin) / (h * w)
        assert area_pct <= 0.70
