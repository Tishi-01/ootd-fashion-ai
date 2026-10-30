# OOTD Fashion AI — Garment & Outfit Detector (Step 1)

A streamlined garment, outerwear layer, footwear, and accessory detector powered by **Google Gemini Vision**:
- **Gemini Vision Garment & Accessory Detection**: Marks every visible garment, layer (`inner`, `mid`, `outer`, `base`), footwear, and accessory with bounding boxes and segmentation masks.
- **Complete Fashion Metadata**: Extracts and enriches structured metadata (`category`, `subcategory`, `color` & CIELAB `shade`/`palette`, `pattern`, `material`, `fit_type`, `silhouette`, `length`, `formality`, `occasions`, `seasons`, `weather`, and `style_tags`) ready for downstream use.
- **Ghost Effect Photo & Ghost Cropped Fallback**:
  - If the photo's ghost effect image already exists in `static/ghost_assets/`, displays that pre-existing ghost effect photo (`ghost_asset`).
  - Otherwise, automatically generates and displays the 3D studio **ghost cropped photo** (`ghost_cropped`).
- **Group Photo & No-Person Photo Rejection**:
  - Automatically checks person count via Gemini Vision and rejects group photos (`person_count > 1`) or photos with no person (`person_count == 0`) with `"Can't detect such photo"`.

---

## 📁 Project Structure

- `api.py`: FastAPI server (`GET /`, `GET /health`, `POST /analyze-outfit`, `GET /api/debug/{image_name}`).
- `pipeline.py`: Unified outfit analysis pipeline coordinating Gemini Vision detection, mask extraction, ghost image selection/cropping, CIELAB color clustering, and rules enrichment.
- `grounded_detector.py`: Google Gemini Vision client, person count validation, GrabCut mask segmentation, and layer subtraction.
- `gemini_ghost.py`: Serves pre-existing ghost effect photos from `static/ghost_assets/` or falls back to ghost cropped photos.
- `ghost_mannequin.py`: 3D hollow ghost cropped image generator (skin removal, inpainting, collar/waistband interior shading, studio drop shadow).
- `color_extractor.py`: CIELAB perceptual color & shade extractor on eroded garment masks.
- `rules.py` & `rules.yaml`: Deterministic fashion rules engine for formality, occasions, seasons, weather, and style tags.
- `taxonomy.py`: Fashion taxonomy definitions (categories, subcategories, patterns, fits, materials, occasions, styles).
- `static/`: Single-page Web UI (`static/index.html`) and pre-rendered ghost assets (`static/ghost_assets/`).
- `tests/`: Automated `pytest` test suite.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run test suite
PYTHONPATH=. pytest -v tests/

# 3. Start Web UI & API server
uvicorn api:app --host 127.0.0.1 --port 8000
```
