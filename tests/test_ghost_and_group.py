import os
import pytest
from pathlib import Path
from pipeline import WardrobePipeline
from ghost_mannequin import create_ghost_mannequin
import numpy as np


@pytest.fixture(scope="module")
def pipeline():
    return WardrobePipeline.get_instance()


def test_ghost_mannequin_unit():
    """Tests the ghost cropped rendering algorithm directly."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[40:160, 40:160] = [180, 50, 50]
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[40:160, 40:160] = 255

    studio_rgb, trans_rgba = create_ghost_mannequin(img, mask, output_size=(400, 400))

    assert studio_rgb.shape == (400, 400, 3)
    assert trans_rgba.shape == (400, 400, 4)
    assert np.all(studio_rgb[0, 0] >= 240)
    assert not np.all(studio_rgb[200, 200] == 255)
    assert trans_rgba[0, 0, 3] == 0


def test_group_photo_rejection(pipeline):
    """Verifies that group photos are rejected with 'Can't detect such photo'."""
    group_path = "tests/images/group_test.jpg"
    assert os.path.exists(group_path), "group_test.jpg not found"

    res = pipeline.analyze_outfit(group_path)
    assert res.get("can_detect") is False
    assert res.get("is_group_photo") is True
    assert "can't detect such photo" in res.get("message", "").lower()
    assert res.get("total_detected") == 0
    assert len(res.get("detected_items", [])) == 0
    assert res.get("status") == "REJECTED_GROUP_PHOTO"


def test_no_person_photo_rejection(pipeline, monkeypatch):
    """Verifies that a photo with no person present is rejected with 'Can't detect such photo'."""
    blank_img = np.ones((300, 300, 3), dtype=np.uint8) * 240

    def fake_detect_outfit(img_rgb, img_bytes=None, api_key=None):
        return {
            "person_count": 0,
            "is_group_photo": False,
            "has_no_person": True,
            "rejection_reason": "Can't detect such photo: No person is present in this photo. Please upload a photo with a single person.",
            "items": [],
        }, "gemini"

    monkeypatch.setattr(pipeline.detector, "detect_outfit", fake_detect_outfit)
    res = pipeline.analyze_outfit(blank_img, img_filename="empty_room.jpg")
    assert res.get("can_detect") is False
    assert res.get("has_no_person") is True
    assert "can't detect such photo" in res.get("message", "").lower()
    assert res.get("total_detected") == 0
    assert res.get("status") == "REJECTED_NO_PERSON"


def test_existing_ghost_asset_vs_ghost_cropped(pipeline):
    """
    Verifies:
    1. When pre-existing ghost effect photo exists in static/ghost_assets/ (e.g. img1.jpg), ghost_source == 'ghost_asset'.
    2. When no pre-existing ghost asset exists for a garment, falls back to ghost cropped photo (ghost_source == 'ghost_cropped').
    """
    res = pipeline.analyze_outfit("tests/images/img1.jpg")
    assert res.get("can_detect") is True
    assert res.get("total_detected") == 5

    for item in res["detected_items"]:
        ghost_url = item.get("ghost_mannequin_url")
        assert ghost_url is not None and ghost_url.startswith("/static/ghost/")
        local_path = Path("." + ghost_url)
        assert local_path.exists()
        assert item.get("ghost_source") == "ghost_asset"
        assert item.get("category") in ["top", "bottom", "overtop", "dress", "footwear", "accessory"]
        assert item.get("subcategory") is not None
        assert item.get("pattern") is not None
        assert item.get("material") is not None
        assert item.get("fit_type") is not None
        assert item.get("formality") is not None
        assert item.get("layer") in ["inner", "mid", "outer", "base"]
        assert isinstance(item.get("style_tags"), list)
        assert isinstance(item.get("occasion", {}).get("types"), list)

    # Also verify ghost_cropped fallback when no pre-existing asset matches
    from gemini_ghost import generate_gemini_ghost_packshot
    img = np.full((200, 200, 3), 120, dtype=np.uint8)
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[30:170, 30:170] = 255
    crop = img[30:170, 30:170]
    fallback_res = generate_gemini_ghost_packshot(
        img_rgb=img,
        mask=mask,
        crop_rgb=crop,
        item_meta={"category": "top", "subcategory": "kurta"},
        output_dir=pipeline.ghost_dir,
        item_id="custom_test",
        image_stem="unseen_user_photo",
        image_hash="0000",
    )
    assert fallback_res["source"] == "ghost_cropped"
    assert Path(fallback_res["ghost_jpg_path"]).exists()
