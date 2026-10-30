import pytest
from rules import FashionRulesEngine

def test_rules_loading():
    engine = FashionRulesEngine.get_instance()
    assert "subcategories" in engine.rules
    assert len(engine.rules["subcategories"]) > 20

def test_kurti_derivation():
    engine = FashionRulesEngine.get_instance()
    meta = engine.derive_attributes(
        category="top",
        subcategory="kurti",
        material="cotton",
        fit="regular",
        color="white",
        shade="pure_white"
    )
    assert meta["formality"] in ["casual", "smart_casual"]
    assert "summer" in meta["seasons"]
    assert "ethnic" in meta["style_tags"]
    assert meta["layering"] == "base"

def test_saree_derivation():
    engine = FashionRulesEngine.get_instance()
    meta = engine.derive_attributes(
        category="dress",
        subcategory="saree",
        material="silk",
        fit="regular",
        color="red",
        shade="red"
    )
    assert meta["formality"] in ["formal", "semi_formal"]
    assert "wedding" in meta["occasions"]
    assert "festival" in meta["occasions"]
    assert "ethnic" in meta["style_tags"]

def test_dupatta_layering():
    engine = FashionRulesEngine.get_instance()
    meta = engine.derive_attributes(
        category="overtop",
        subcategory="dupatta_stole",
        material="chiffon_georgette",
        fit="loose",
        color="pink",
        shade="baby_pink"
    )
    assert meta["layering"] == "outer"
