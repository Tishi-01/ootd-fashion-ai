"""
OOTD Fashion AI — Garment Detector, Digital Wardrobe & 4-Pillar ANN Recommender API.
Provides:
1. Gemini Vision Garment & Accessory Detection (`/analyze-outfit`)
2. User Profile with Date of Birth & Location (`/api/profile`)
3. Digital Wardrobe Store with Duplicate Detection (`/api/wardrobe/*`)
4. Live Weather & Planetary Astrology Lucky Color Calculators (`/api/weather`, `/api/astrology`)
5. 4-Pillar PyTorch ANN + ML Outfit Recommendation Engine (`/api/recommend`)
"""

import io
import cv2
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query, Body
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline import WardrobePipeline
from wardrobe_store import (
    get_profile,
    update_profile,
    list_wardrobe_items,
    add_item_to_wardrobe,
    delete_wardrobe_item,
    clear_wardrobe,
    seed_predefined_wardrobe,
)
from recommender import (
    NeuralOutfitRecommender,
    fetch_today_weather,
    calculate_astrology_lucky_colors,
)

pipeline: Optional[WardrobePipeline] = None
recommender: Optional[NeuralOutfitRecommender] = None


def get_pipeline() -> WardrobePipeline:
    global pipeline
    if pipeline is None:
        pipeline = WardrobePipeline.get_instance()
    return pipeline


def get_recommender() -> NeuralOutfitRecommender:
    global recommender
    if recommender is None:
        recommender = NeuralOutfitRecommender.get_instance()
    return recommender


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[API] Initializing OOTD Gemini Vision Detector & 4-Pillar ANN Recommender...")
    if len(list_wardrobe_items()) == 0:
        seed_predefined_wardrobe()
    get_pipeline()
    get_recommender()
    print("[API] OOTD Fashion AI online and ready.")
    yield


app = FastAPI(
    title="OOTD Fashion AI — Detector, Wardrobe & 4-Pillar ANN Recommender",
    description="Gemini Vision Garment Detector + Digital Wardrobe + 4-Pillar PyTorch ANN Recommendation Engine",
    version="4.0.0",
    lifespan=lifespan,
)

# Mount static & debug directories
static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

debug_dir = Path(__file__).resolve().parent / "debug"
debug_dir.mkdir(parents=True, exist_ok=True)
app.mount("/debug", StaticFiles(directory=str(debug_dir)), name="debug")

tests_images_dir = Path(__file__).resolve().parent / "tests" / "images"
if tests_images_dir.exists():
    app.mount("/tests/images", StaticFiles(directory=str(tests_images_dir)), name="tests_images")


def decode_image_bytes(contents: bytes) -> Optional[np.ndarray]:
    try:
        pil_img = Image.open(io.BytesIO(contents))
        pil_img = ImageOps.exif_transpose(pil_img)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        max_dim = 1600
        w, h = pil_img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        return np.array(pil_img)
    except Exception:
        nparr = np.frombuffer(contents, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------
class ProfileUpdateRequest(BaseModel):
    name: Optional[str] = None
    dob: Optional[str] = None
    city: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class RecommendRequest(BaseModel):
    use_weather: bool = False
    use_occasion: bool = False
    use_astrology: bool = False
    use_color_fit_harmony: bool = True  # Always ON
    occasion: str = "casual_outing"
    city: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    override_temp_c: Optional[float] = None
    dob: Optional[str] = None
    top_k: int = 4


# ---------------------------------------------------------------------------
# Core & Detector Endpoints
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    html_path = static_dir / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    return HTMLResponse("<h1>OOTD Fashion AI Online</h1><p>Visit <a href='/docs'>/docs</a></p>")


@app.get("/health")
def health():
    items = list_wardrobe_items()
    return {
        "status": "online",
        "version": "4.0.0",
        "primary_detector": "Google Gemini Vision (google.genai)",
        "segmentation": "Promptable Layered GrabCut + Skin Filter",
        "color_engine": "CIELAB Hue-Aware Perceptual Clustering",
        "recommender": "4-Pillar PyTorch ANN + ML Gradient Boosting Engine",
        "wardrobe_count": len(items),
    }


@app.post("/analyze-outfit")
async def analyze_outfit(
    file: UploadFile = File(...),
    api_key: Optional[str] = Form(None),
):
    """
    Analyzes an outfit photo using Gemini Vision:
    - If group photo or no person present: returns can_detect=False and message="Can't detect such photo..."
    - If single person: marks garments & accessories, fills metadata, and returns ghost effect or ghost cropped photos.
    """
    try:
        contents = await file.read()
        if not contents:
            return JSONResponse(status_code=400, content={"error": "Empty file.", "garments": []})

        img_rgb = decode_image_bytes(contents)
        if img_rgb is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image format.", "garments": []})

        res = get_pipeline().analyze_outfit(
            img_rgb,
            raw_bytes=contents,
            img_filename=file.filename,
            api_key=api_key,
        )

        if not res.get("can_detect", True) or res.get("is_group_photo") or res.get("has_no_person"):
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "can_detect": False,
                    "is_group_photo": bool(res.get("is_group_photo")),
                    "has_no_person": bool(res.get("has_no_person")),
                    "person_count": res.get("person_count", 0),
                    "message": res.get(
                        "message",
                        "Can't detect such photo: Please upload a photo with a single person.",
                    ),
                    "garments": [],
                    "count": 0,
                    "engine": res.get("engine", "gemini"),
                    "status": res.get("status", "REJECTED_PHOTO"),
                },
            )

        debug_url = f"/debug/{Path(res['debug_overlay_path']).name}" if res.get("debug_overlay_path") else None
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "can_detect": True,
                "is_group_photo": False,
                "has_no_person": False,
                "person_count": 1,
                "garments": res["detected_items"],
                "count": res["total_detected"],
                "engine": res.get("engine", "gemini"),
                "debug_overlay_url": debug_url,
                "status": "OK",
            },
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "garments": []})


# ---------------------------------------------------------------------------
# User Profile, Live Weather & Astrology Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/profile")
def api_get_profile():
    prof = get_profile()
    astro = calculate_astrology_lucky_colors(prof.get("dob"))
    return {
        "status": "OK",
        "profile": prof,
        "astrology": astro,
    }


@app.post("/api/profile")
def api_update_profile(req: ProfileUpdateRequest):
    prof = update_profile(
        name=req.name,
        dob=req.dob,
        city=req.city,
        latitude=req.latitude,
        longitude=req.longitude,
    )
    astro = calculate_astrology_lucky_colors(prof.get("dob"))
    return {
        "status": "OK",
        "profile": prof,
        "astrology": astro,
    }


@app.get("/api/weather")
def api_get_weather(
    city: Optional[str] = Query(None),
    lat: Optional[float] = Query(None),
    lon: Optional[float] = Query(None),
    temp: Optional[float] = Query(None),
):
    prof = get_profile()
    effective_city = city or prof.get("city") or "Mumbai"
    w = fetch_today_weather(
        city=effective_city,
        latitude=lat,
        longitude=lon,
        override_temp_c=temp,
    )
    return {"status": "OK", "weather": w}


@app.get("/api/astrology")
def api_get_astrology(dob: Optional[str] = Query(None)):
    prof = get_profile()
    effective_dob = dob or prof.get("dob") or "2003-05-15"
    astro = calculate_astrology_lucky_colors(effective_dob)
    return {"status": "OK", "astrology": astro}


# ---------------------------------------------------------------------------
# Digital Wardrobe CRUD with Duplicate Detection
# ---------------------------------------------------------------------------
@app.get("/api/wardrobe")
def api_list_wardrobe(category: Optional[str] = Query(None)):
    items = list_wardrobe_items(category=category)
    return {
        "status": "OK",
        "count": len(items),
        "items": items,
    }


@app.post("/api/wardrobe/add")
def api_add_wardrobe_item(item_data: Dict[str, Any] = Body(...)):
    item, is_dup, msg = add_item_to_wardrobe(item_data)
    return {
        "status": "OK",
        "is_duplicate": is_dup,
        "message": msg,
        "item": item,
    }


@app.post("/api/wardrobe/add-batch")
def api_add_wardrobe_batch(items: List[Dict[str, Any]] = Body(...)):
    added = []
    duplicates = []
    for it in items:
        saved, is_dup, msg = add_item_to_wardrobe(it)
        if is_dup:
            duplicates.append({"item": saved, "message": msg})
        else:
            added.append({"item": saved, "message": msg})
    return {
        "status": "OK",
        "added_count": len(added),
        "duplicate_count": len(duplicates),
        "added": added,
        "duplicates": duplicates,
    }


@app.delete("/api/wardrobe/{item_id}")
def api_delete_wardrobe_item(item_id: int):
    ok = delete_wardrobe_item(item_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Item not found in wardrobe.")
    return {"status": "OK", "message": f"Deleted item #{item_id}."}


@app.delete("/api/wardrobe")
def api_clear_wardrobe():
    clear_wardrobe()
    return {"status": "OK", "message": "Wardrobe cleared."}


@app.post("/api/wardrobe/seed")
def api_seed_predefined_wardrobe():
    res = seed_predefined_wardrobe()
    return {
        "status": "OK",
        "message": f"Loaded predefined wardrobe ({res['added_count']} added, {res['total_wardrobe']} total).",
        **res,
    }


# ---------------------------------------------------------------------------
# 4-Pillar PyTorch ANN + ML Outfit Recommendation Endpoint
# ---------------------------------------------------------------------------
@app.post("/api/recommend")
def api_recommend_outfits(req: RecommendRequest):
    prof = get_profile()
    wardrobe_items = list_wardrobe_items()

    effective_city = req.city or prof.get("city") or "Mumbai"
    effective_dob = req.dob or prof.get("dob") or "2003-05-15"

    res = get_recommender().recommend_outfits(
        wardrobe_items=wardrobe_items,
        use_weather=req.use_weather,
        use_occasion=req.use_occasion,
        use_astrology=req.use_astrology,
        use_color_fit_harmony=True,  # Always ON
        occasion=req.occasion,
        dob=effective_dob,
        city=effective_city,
        latitude=req.latitude,
        longitude=req.longitude,
        override_temp_c=req.override_temp_c,
        top_k=req.top_k,
    )
    return res


@app.get("/api/debug/{image_name}")
def get_debug_image(image_name: str):
    debug_path = Path(__file__).resolve().parent / "debug" / image_name
    if debug_path.exists():
        return FileResponse(str(debug_path), media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Debug image not found.")


if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
