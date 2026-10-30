from fastapi.testclient import TestClient
import wardrobe_store
from recommender import evaluate_sartorial_compatibility
from api import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "Gemini Vision" in data["primary_detector"]
    assert "PyTorch" in data["recommender"]


def test_analyze_single_person_api():
    with open("tests/images/img1.jpg", "rb") as f:
        res = client.post("/analyze-outfit", files={"file": ("img1.jpg", f, "image/jpeg")})
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["can_detect"] is True
    assert data["is_group_photo"] is False
    assert data["has_no_person"] is False
    assert data["count"] == 5
    assert len(data["garments"]) == 5


def test_analyze_group_photo_rejected_api():
    with open("tests/images/group_test.jpg", "rb") as f:
        res = client.post("/analyze-outfit", files={"file": ("group_test.jpg", f, "image/jpeg")})
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is False
    assert data["can_detect"] is False
    assert data["is_group_photo"] is True
    assert "can't detect such photo" in data["message"].lower()
    assert data["count"] == 0


def test_profile_and_astrology_lucky_colors(tmp_path, monkeypatch):
    monkeypatch.setattr(wardrobe_store, "DB_PATH", tmp_path / "test_wardrobe.db")
    wardrobe_store.init_db(seed_if_empty=False)

    res = client.post(
        "/api/profile",
        json={"name": "Tanishk", "dob": "2003-05-15", "city": "Mumbai"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "OK"
    assert data["profile"]["dob"] == "2003-05-15"
    assert data["astrology"]["birth_number"] == 6
    assert "Venus" in data["astrology"]["birth_planet"]
    assert len(data["astrology"]["lucky_shades"]) >= 2
    assert len(data["astrology"]["display_swatches"]) >= 2


def test_weather_endpoint_regimes():
    hot_res = client.get("/api/weather", params={"city": "Mumbai", "temp": 33.0})
    assert hot_res.status_code == 200
    assert hot_res.json()["weather"]["regime"] == "hot"

    cold_res = client.get("/api/weather", params={"city": "Delhi", "temp": 10.0})
    assert cold_res.status_code == 200
    assert cold_res.json()["weather"]["regime"] == "cold"


def test_sartorial_pairing_rules_block_invalid_combinations():
    tshirt = {"category": "top", "subcategory": "tshirt", "formality": "casual", "pattern": "plain", "fit_type": "regular"}
    shirt = {"category": "top", "subcategory": "shirt", "formality": "formal", "pattern": "plain", "fit_type": "tailored"}
    kurta = {"category": "top", "subcategory": "kurta", "formality": "semi_formal", "pattern": "embroidered", "fit_type": "regular"}

    churidar = {"category": "bottom", "subcategory": "churidar", "formality": "formal"}
    pajama = {"category": "bottom", "subcategory": "pajama", "formality": "semi_formal"}
    trousers = {"category": "bottom", "subcategory": "trousers", "formality": "formal"}
    jeans = {"category": "bottom", "subcategory": "jeans", "formality": "casual"}
    cargos = {"category": "bottom", "subcategory": "cargos", "formality": "casual"}

    coat = {"category": "overtop", "subcategory": "coat", "formality": "formal"}
    biker_jacket = {"category": "overtop", "subcategory": "jacket", "formality": "casual"}
    nehru_jacket = {"category": "overtop", "subcategory": "nehru_jacket", "formality": "semi_formal"}

    mojari = {"category": "footwear", "subcategory": "mojari", "formality": "semi_formal"}
    sneakers = {"category": "footwear", "subcategory": "sneakers", "formality": "casual"}
    oxford = {"category": "footwear", "subcategory": "oxford", "formality": "formal"}

    # 1. T-Shirt with Churidar or Pajama -> MUST BE BLOCKED
    assert evaluate_sartorial_compatibility([tshirt, churidar])[0] is False
    assert evaluate_sartorial_compatibility([tshirt, pajama])[0] is False
    assert evaluate_sartorial_compatibility([shirt, churidar])[0] is False

    # 2. Western Coat or Biker Jacket with Kurta -> MUST BE BLOCKED
    assert evaluate_sartorial_compatibility([kurta, pajama, coat])[0] is False
    assert evaluate_sartorial_compatibility([kurta, churidar, biker_jacket])[0] is False
    assert evaluate_sartorial_compatibility([kurta, cargos])[0] is False

    # 3. Traditional Kurta + Churidar with Sneakers or Oxfords -> MUST BE BLOCKED
    assert evaluate_sartorial_compatibility([kurta, churidar, sneakers])[0] is False
    assert evaluate_sartorial_compatibility([kurta, churidar, oxford])[0] is False

    # 4. Valid Canonical Combinations -> MUST BE APPROVED WITH HIGH HARMONY
    valid_ethnic, tb_s, ls_s, _ = evaluate_sartorial_compatibility([kurta, churidar, nehru_jacket, mojari])
    assert valid_ethnic is True
    assert tb_s == 1.0 and ls_s == 1.0

    valid_formal, tb_f, ls_f, _ = evaluate_sartorial_compatibility([shirt, trousers, coat, oxford])
    assert valid_formal is True
    assert tb_f == 1.0 and ls_f >= 0.95

    valid_casual, tb_c, ls_c, _ = evaluate_sartorial_compatibility([tshirt, jeans, biker_jacket, sneakers])
    assert valid_casual is True
    assert tb_c == 1.0 and ls_c >= 0.95


def test_predefined_wardrobe_and_4pillar_recommender(tmp_path, monkeypatch):
    monkeypatch.setattr(wardrobe_store, "DB_PATH", tmp_path / "test_wardrobe.db")
    wardrobe_store.init_db(seed_if_empty=True)

    w_res = client.get("/api/wardrobe")
    assert w_res.status_code == 200
    assert w_res.json()["count"] == len(wardrobe_store.PREDEFINED_WARDROBE_CATALOG)

    # Duplicate check when adding an already-seeded item
    dup_check = client.post("/api/wardrobe/add", json=wardrobe_store.PREDEFINED_WARDROBE_CATALOG[0])
    assert dup_check.status_code == 200
    assert dup_check.json()["is_duplicate"] is True

    # 1. Festival / Ethnic recommendation -> Must recommend Kurta + Pajama/Churidar (+ Loafers/Sandals), NEVER T-Shirt + Churidar or Coat + Kurta
    rec_fest = client.post(
        "/api/recommend",
        json={
            "use_weather": True,
            "use_occasion": True,
            "occasion": "festival",
            "use_color_fit_harmony": True,
            "use_astrology": True,
            "override_temp_c": 26.0,
        },
    )
    assert rec_fest.status_code == 200
    fest_top = rec_fest.json()["recommendations"][0]
    fest_subcats = {it["category"]: it["subcategory"] for it in fest_top["items"]}
    assert fest_subcats["top"] == "kurta"
    assert fest_subcats["bottom"] in {"pajama", "churidar"}
    assert "overtop" not in fest_subcats or fest_subcats["overtop"] == "nehru_jacket"
    assert fest_subcats.get("footwear") in {"mojari", "loafers", "sandals"}

    # 2. Work / Formal recommendation in Hot weather -> Must recommend Shirt + Trousers/Chinos, no outerwear
    rec_work = client.post(
        "/api/recommend",
        json={
            "use_weather": True,
            "use_occasion": True,
            "occasion": "work_formal",
            "use_color_fit_harmony": True,
            "use_astrology": True,
            "override_temp_c": 32.0,
        },
    )
    assert rec_work.status_code == 200
    work_top = rec_work.json()["recommendations"][0]
    work_subcats = {it["category"]: it["subcategory"] for it in work_top["items"]}
    assert work_subcats["top"] == "shirt"
    assert work_subcats["bottom"] in {"trousers", "chinos"}
    assert "overtop" not in work_subcats

    # 3. Casual Outing / College in Cold weather -> Must recommend Casual Top + Jeans/Chinos/Cargos + Jacket/Blazer, never Churidar/Pajama
    rec_casual = client.post(
        "/api/recommend",
        json={
            "use_weather": True,
            "use_occasion": True,
            "occasion": "college",
            "use_color_fit_harmony": True,
            "use_astrology": False,
            "override_temp_c": 11.0,
        },
    )
    assert rec_casual.status_code == 200
    casual_recs = rec_casual.json()["recommendations"]
    assert len(casual_recs) == 4
    casual_tops = {next(it["id"] for it in r["items"] if it["category"] == "top") for r in casual_recs}
    casual_bots = {next(it["id"] for it in r["items"] if it["category"] == "bottom") for r in casual_recs}
    assert len(casual_tops) == 4
    assert len(casual_bots) >= 3
    for rec in casual_recs:
        subcats = {it["category"]: it["subcategory"] for it in rec["items"]}
        assert subcats["bottom"] not in {"churidar", "pajama"}
        if subcats["top"] == "kurta":
            assert subcats.get("overtop") not in {"coat", "blazer", "jacket"}

    # 4. Gym / Sports recommendation -> Must recommend T-Shirt + Joggers + Running Trainers/Sneakers
    rec_gym = client.post(
        "/api/recommend",
        json={
            "use_weather": True,
            "use_occasion": True,
            "occasion": "gym_sports",
            "use_color_fit_harmony": True,
            "use_astrology": True,
            "override_temp_c": 24.0,
        },
    )
    assert rec_gym.status_code == 200
    gym_recs = rec_gym.json()["recommendations"]
    assert len(gym_recs) == 4
    for r in gym_recs:
        subcats = {it["category"]: it["subcategory"] for it in r["items"]}
        assert subcats["top"] == "tshirt"
        assert subcats["bottom"] == "joggers"
        assert subcats["footwear"] in {"sports_shoes", "sneakers"}

    # 5. Verify 100% Men's Wardrobe (no handbag, women's wide trousers, or women's strappy sandals)
    slugs = {it["slug"] for it in wardrobe_store.PREDEFINED_WARDROBE_CATALOG}
    assert "olive_leather_handbag" not in slugs
    assert "tan_strappy_sandals" not in slugs
    assert "ivory_wide_trousers" not in slugs
    assert "grey_plaid_blazer" not in slugs
