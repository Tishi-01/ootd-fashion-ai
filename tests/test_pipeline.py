import os
import pytest
from pipeline import WardrobePipeline


@pytest.fixture(scope="module")
def pipeline():
    return WardrobePipeline.get_instance()


def test_pipeline_initialization(pipeline):
    assert pipeline is not None
    assert pipeline.detector is not None
    assert pipeline.rules_engine is not None


def test_pipeline_outfit_analysis(pipeline):
    img_path = "tests/images/img2.jpg"
    assert os.path.exists(img_path)

    res = pipeline.analyze_outfit(img_path)
    assert res["status"] == "OK"
    assert res["can_detect"] is True
    assert res["total_detected"] == 3

    categories = [it["category"] for it in res["detected_items"]]
    assert "overtop" in categories
    assert "top" in categories
    assert "bottom" in categories

    for it in res["detected_items"]:
        assert "id" in it
        assert "name" in it
        assert "category" in it
        assert "subcategory" in it
        assert "color" in it
        assert "confidence" in it
        assert "image_url" in it
        assert "mask_url" in it
        assert "ghost_mannequin_url" in it
        crop_path = os.path.join(pipeline.static_dir, it["image_url"].replace("/static/", ""))
        mask_path = os.path.join(pipeline.static_dir, it["mask_url"].replace("/static/", ""))
        ghost_path = os.path.join(pipeline.static_dir, it["ghost_mannequin_url"].replace("/static/", ""))
        assert os.path.exists(crop_path)
        assert os.path.exists(mask_path)
        assert os.path.exists(ghost_path)
