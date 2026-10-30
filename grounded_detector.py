"""
Grounded Garment & Accessory Detector powered by Google Gemini Vision.
1. Validates person presence:
   - Rejects group photos (person_count > 1) with "Can't detect such photo".
   - Rejects photos with no person present (person_count == 0) with "Can't detect such photo".
2. For single-person photos (person_count == 1):
   - Detects every visible garment, outerwear layer (inner/mid/outer/base), footwear, and accessory.
   - Extracts bounding boxes and complete fashion metadata.
3. Generates clean garment segmentation masks with GrabCut, layer subtraction, and skin removal.
"""

import os
import cv2
import json
import hashlib
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field

import taxonomy
from ghost_mannequin import extract_skin_mask

logger = logging.getLogger("grounded_detector")
logger.setLevel(logging.INFO)

CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "llm_vision_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _load_env_api_key() -> Optional[str]:
    key = os.environ.get("GEMINI_API_KEY")
    if key and key.strip():
        return key.strip()
    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("GEMINI_API_KEY="):
                val = line.split("=", 1)[1].strip()
                if val:
                    return val
    return None


class GroundedItem(BaseModel):
    box_2d: List[float] = Field(description="Bounding box [ymin, xmin, ymax, xmax] in 0..1 or 0..1000")
    name: str = Field(default="Garment", description="Descriptive garment name (e.g. 'Charcoal Checkered Blazer')")
    label: str = Field(default="Garment", description="Concise human-readable label")
    category: str = Field(default="top", description="One of: top, bottom, overtop, dress, footwear, accessory")
    subcategory: str = Field(default="tshirt", description="Specific garment or accessory subcategory from taxonomy")
    pattern: str = Field(default="plain", description="plain, striped, checkered, graphic, floral, polka_dot, distressed, colorblock, tie_dye")
    fit: str = Field(default="regular", description="oversized, slim, regular, skinny, baggy, tailored, loose")
    silhouette: str = Field(default="straight", description="relaxed, fitted, boxy, straight, flared, a_line, tapered")
    length: str = Field(default="regular", description="crop, regular, longline, knee_length, ankle_length, mini, maxi")
    material: str = Field(default="cotton", description="cotton, denim, polyester, linen, leather, wool, silk, nylon, knit, corduroy, chiffon_georgette, velvet, tweed")
    formality: str = Field(default="casual", description="casual, smart_casual, semi_formal, formal")
    color_primary: str = Field(default="white", description="Primary color shade name")
    color_secondary: Optional[str] = Field(default=None, description="Secondary color shade name if any")
    layer: str = Field(default="base", description="inner, mid, outer, or base")
    style_tags: List[str] = Field(default_factory=lambda: ["casual"], description="Style tags list")
    occasions: List[str] = Field(default_factory=lambda: ["casual_outing"], description="Suitable occasions list")
    seasons: List[str] = Field(default_factory=lambda: ["summer", "spring"], description="Suitable seasons list")
    weather: List[str] = Field(default_factory=lambda: ["warm"], description="Suitable weather conditions list")
    confidence: float = Field(default=0.95, ge=0.0, le=1.0, description="Detection confidence score")


class GroundedOutfitResponse(BaseModel):
    person_count: int = Field(default=1, description="Number of distinct human persons visible in the photo (0, 1, 2, ...)")
    is_group_photo: bool = Field(default=False, description="True if 2 or more people are present in the photo")
    has_no_person: bool = Field(default=False, description="True if 0 people are present in the photo")
    rejection_reason: Optional[str] = Field(default=None, description="Reason if photo cannot be detected (group photo or no person)")
    items: List[GroundedItem] = Field(default_factory=list, description="Detected garments, footwear, and accessories (empty if rejected)")


def normalize_box(box: List[float], h: int, w: int) -> Tuple[List[float], List[int]]:
    """
    Converts box_2d to normalized [ymin, xmin, ymax, xmax] in 0..1
    and integer pixel coordinates [ymin, xmin, ymax, xmax].
    """
    b = list(box) if box and len(box) == 4 else [0.1, 0.1, 0.9, 0.9]
    if any(c > 1.0 for c in b):
        norm = [
            max(0.0, min(1.0, b[0] / 1000.0)),
            max(0.0, min(1.0, b[1] / 1000.0)),
            max(0.0, min(1.0, b[2] / 1000.0)),
            max(0.0, min(1.0, b[3] / 1000.0)),
        ]
    else:
        norm = [
            max(0.0, min(1.0, b[0])),
            max(0.0, min(1.0, b[1])),
            max(0.0, min(1.0, b[2])),
            max(0.0, min(1.0, b[3])),
        ]

    ymin_n, xmin_n, ymax_n, xmax_n = norm
    if ymin_n > ymax_n:
        ymin_n, ymax_n = ymax_n, ymin_n
    if xmin_n > xmax_n:
        xmin_n, xmax_n = xmax_n, xmin_n

    ymin = int(round(ymin_n * h))
    xmin = int(round(xmin_n * w))
    ymax = max(ymin + 5, int(round(ymax_n * h)))
    xmax = max(xmin + 5, int(round(xmax_n * w)))

    ymin = max(0, min(h - 1, ymin))
    xmin = max(0, min(w - 1, xmin))
    ymax = max(ymin + 5, min(h, ymax))
    xmax = max(xmin + 5, min(w, xmax))

    return [round(ymin_n, 4), round(xmin_n, 4), round(ymax_n, 4), round(xmax_n, 4)], [ymin, xmin, ymax, xmax]


def _sanitize_item_metadata(raw_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes and guarantees complete, valid taxonomy metadata for every detected garment or accessory.
    """
    item = dict(raw_item)
    subcat = str(item.get("subcategory") or "tshirt").strip().lower().replace(" ", "_").replace("-", "_")
    cat = str(item.get("category") or "top").strip().lower()

    # Synonym mapping for common Gemini outputs
    subcat_synonyms = {
        "t_shirt": "tshirt",
        "tee": "tshirt",
        "graphic_tee": "tshirt",
        "suit_jacket": "blazer",
        "coat": "trench_coat",
        "overcoat": "trench_coat",
        "bomber": "bomber_jacket",
        "pants": "trousers",
        "slacks": "trousers",
        "dress_pants": "formal_trousers",
        "wide_leg_pants": "wide_leg_trousers",
        "trackpants": "track_pants",
        "sweat_pants": "sweatpants",
        "shoes": "sneakers",
        "trainers": "sneakers",
        "dress_shoes": "formal_shoes",
        "oxfords": "formal_shoes",
        "bag": "handbag",
        "purse": "handbag",
        "tote": "handbag",
        "tote_bag": "handbag",
        "clutch": "handbag",
        "shades": "sunglasses",
        "eyewear": "sunglasses",
        "glasses": "sunglasses",
        "hat": "cap",
        "baseball_cap": "cap",
        "beanie": "cap",
        "wristwatch": "watch",
        "bangle": "bracelet",
        "dupatta": "dupatta_stole",
        "stole": "dupatta_stole",
        "juttis": "juttis_mojaris",
        "mojaris": "juttis_mojaris",
    }
    if subcat in subcat_synonyms:
        subcat = subcat_synonyms[subcat]

    if subcat in taxonomy.SUBCAT_TO_PARENT:
        cat = taxonomy.SUBCAT_TO_PARENT[subcat]
    elif cat not in taxonomy.CATEGORIES:
        cat = "top"

    item["category"] = cat
    item["subcategory"] = subcat

    # Default layer by category if missing
    layer = str(item.get("layer") or "base").strip().lower()
    if layer not in ["inner", "mid", "outer", "base"]:
        layer = "outer" if cat == "overtop" else "base"
    item["layer"] = layer

    # Ensure pattern, fit, silhouette, length, material, formality are populated
    item["pattern"] = str(item.get("pattern") or "plain").strip().lower().replace(" ", "_")
    if item["pattern"] == "plaid":
        item["pattern"] = "checkered"
    if item["pattern"] == "solid":
        item["pattern"] = "plain"

    item["fit"] = str(item.get("fit") or "regular").strip().lower().replace(" ", "_")
    item["silhouette"] = str(item.get("silhouette") or "straight").strip().lower().replace(" ", "_")
    item["length"] = str(item.get("length") or "regular").strip().lower().replace(" ", "_")
    item["material"] = str(item.get("material") or "cotton").strip().lower().replace(" ", "_")
    item["formality"] = str(item.get("formality") or "casual").strip().lower().replace(" ", "_")
    item["confidence"] = float(item.get("confidence") or 0.95)

    if not item.get("label"):
        item["label"] = f"{item.get('color_primary', '')} {subcat.replace('_', ' ')}".strip()
    if not item.get("name"):
        item["name"] = item["label"].title()

    return item


class GroundedFashionDetector:
    _instance = None

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or _load_env_api_key()

    @classmethod
    def get_instance(cls, api_key: Optional[str] = None):
        if cls._instance is None:
            cls._instance = cls(api_key)
        elif api_key:
            cls._instance.api_key = api_key
        return cls._instance

    def detect_outfit(
        self,
        img_rgb: np.ndarray,
        img_bytes: Optional[bytes] = None,
        api_key: Optional[str] = None
    ) -> Tuple[Dict[str, Any], str]:
        """
        Runs Gemini Vision grounded detection, person presence/count validation, and metadata extraction.
        Rejects photos if:
        - person_count > 1 (group photo)
        - person_count == 0 (no person present)
        """
        effective_key = api_key or self.api_key or _load_env_api_key()

        if img_bytes is None:
            _, buf = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            img_bytes = buf.tobytes()

        candidates = []
        if img_bytes:
            candidates.append(hashlib.sha256(img_bytes).hexdigest())
        candidates.append(hashlib.sha256(img_rgb.tobytes()).hexdigest())
        _, buf = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        candidates.append(hashlib.sha256(buf.tobytes()).hexdigest())

        # 1. Check disk cache
        for hsh in candidates:
            cache_file = CACHE_DIR / f"{hsh}.json"
            if cache_file.exists():
                try:
                    cached_json = cache_file.read_text(encoding="utf-8")
                    data = json.loads(cached_json)
                    parsed = GroundedOutfitResponse.model_validate(data)
                    out = parsed.model_dump()
                    out["items"] = [_sanitize_item_metadata(it) for it in out.get("items", [])]
                    logger.info(
                        f"[GroundedDetector] Loaded from cache ({hsh[:8]}): "
                        f"persons={out['person_count']}, group={out['is_group_photo']}, no_person={out['has_no_person']}, items={len(out['items'])}"
                    )
                    return out, "gemini"
                except Exception as e:
                    logger.warning(f"[GroundedDetector] Failed to parse cached file {cache_file.name}: {e}")

        img_hash = candidates[0] if candidates else hashlib.sha256(img_rgb.tobytes()).hexdigest()
        cache_file = CACHE_DIR / f"{img_hash}.json"

        # 2. Call Gemini Vision API
        if not effective_key:
            raise RuntimeError("GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in .env.")

        resp_data = self._call_gemini_vision(img_rgb, effective_key)
        resp_data["items"] = [_sanitize_item_metadata(it) for it in resp_data.get("items", [])]
        cache_file.write_text(json.dumps(resp_data, indent=2), encoding="utf-8")
        logger.info(f"[GroundedDetector] Cached Gemini Vision response ({img_hash[:8]})")
        return resp_data, "gemini"

    def _call_gemini_vision(self, img_rgb: np.ndarray, api_key: str) -> Dict[str, Any]:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        # Resize image to <= 1024px max dimension for fast, reliable Gemini Vision inference
        h, w = img_rgb.shape[:2]
        scale = min(1024.0 / max(h, w), 1.0)
        img_opt = cv2.resize(img_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else img_rgb
        _, buf = cv2.imencode(".jpg", cv2.cvtColor(img_opt, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        img_part = types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg")

        prompt = f"""
You are an expert fashion AI analyzing photos for a digital wardrobe detector.

STEP 1 — MANDATORY PERSON PRESENCE & GROUP PHOTO CHECK:
Count the number of distinct human people visible in the image (`person_count`).

Case A — NO PERSON PRESENT (`person_count == 0`):
If there is NO human person visible in the photo (e.g., an empty scene, landscape, animal, random object, or standalone/flat-lay clothing without a person wearing it):
- Set "person_count": 0
- Set "has_no_person": true
- Set "is_group_photo": false
- Set "rejection_reason": "Can't detect such photo: No person is present in this photo. Please upload a photo with a single person."
- Set "items": []
Do NOT extract any items when no person is present.

Case B — GROUP PHOTO (`person_count >= 2`):
If there are 2 or more people visible in the photo (e.g., group photo, couple, friends, crowd):
- Set "person_count": <number_of_people>
- Set "is_group_photo": true
- Set "has_no_person": false
- Set "rejection_reason": "Can't detect such photo: Group photos are not allowed. Please upload a photo with a single person."
- Set "items": []
Do NOT extract any items for group photos.

Case C — SINGLE PERSON (`person_count == 1`):
Only when there is EXACTLY 1 person visible in the photo:
- Set "person_count": 1
- Set "is_group_photo": false
- Set "has_no_person": false
- Set "rejection_reason": null
- Detect and extract EVERY visible garment, outerwear layer, footwear, and accessory worn or carried by that person!

STEP 2 — GARMENT, LAYER, FOOTWEAR & ACCESSORY EXTRACTION RULES (for single-person photos):
1. LAYERING:
   - Always separate outer layers from inner layers!
   - If the person wears a blazer, jacket, bomber_jacket, denim_jacket, leather_jacket, cardigan, hoodie, coat, vest, or dupatta_stole over a shirt, tshirt, tank_top, crop_top, or kurti:
     * Extract the outer garment with category="overtop", layer="outer".
     * Extract the inner garment as a SEPARATE item with category="top", layer="inner".
2. TOPS, BOTTOMS & DRESSES:
   - Identify tops (`tshirt`, `shirt`, `polo`, `tank_top`, `crop_top`, `blouse`, `corset`, `sweatshirt`, `sweater`, `kurti`, `kurta`, `saree_blouse`).
   - Identify bottoms (`jeans`, `trousers`, `wide_leg_trousers`, `formal_trousers`, `chinos`, `joggers`, `track_pants`, `shorts`, `skirt`, `cargo_pants`, `sweatpants`, `leggings`, `palazzo`, `salwar`, `churidar`, `dhoti_pants`).
   - Identify one-piece dresses / ethnic ensembles (`casual_dress`, `maxi_dress`, `slip_dress`, `bodycon_dress`, `shirt_dress`, `saree`, `anarkali`, `lehenga`).
3. FOOTWEAR:
   - Extract visible shoes/footwear (`sneakers`, `boots`, `sandals`, `platform_sandals`, `formal_shoes`, `loafers`, `heels`, `flats`, `juttis_mojaris`, `kolhapuris`, `ethnic_flats`) with category="footwear".
4. ACCESSORIES:
   - Extract all clearly visible accessories (`handbag`, `backpack`, `belt`, `watch`, `bracelet`, `sunglasses`, `cap`, `scarf`) with category="accessory".
5. BOUNDING BOX & METADATA:
   - `box_2d`: tight bounding box `[ymin, xmin, ymax, xmax]` in `0..1000` scale around that specific item.
   - `category`: one of {taxonomy.CATEGORIES}
   - `subcategory`: one of {taxonomy.ALL_SUBCATEGORIES}
   - `pattern`: one of {taxonomy.PATTERNS}
   - `fit`: one of {taxonomy.FIT_TYPES}
   - `silhouette`: one of {taxonomy.SILHOUETTES}
   - `length`: one of {taxonomy.LENGTHS}
   - `material`: one of {taxonomy.MATERIALS}
   - `formality`: one of {taxonomy.FORMALITIES}
   - `color_primary`: accurate shade name (e.g., `pure_white`, `off_white`, `black`, `charcoal_grey`, `grey`, `dark_navy`, `navy`, `indigo_denim`, `faded_denim`, `blue`, `olive_green`, `khaki`, `beige`, `tan`, `brown`, `maroon`, `burgundy`, `red`, `pink`, `mustard`, `teal`)
   - `layer`: `inner`, `mid`, `outer`, or `base`
   - `style_tags`: list from {taxonomy.STYLE_TAGS}
   - `occasions`: list from {taxonomy.OCCASIONS}
   - `seasons`: list from {taxonomy.SEASONS}
   - `weather`: list from {taxonomy.CONDITIONS}
"""

        # Ordered by speed and reliability; models with thinking_budget=0 run in ~4-5s
        models_to_try = [
            ("gemini-3.6-flash", True),
            ("gemini-3.5-flash-lite", False),
            ("gemini-flash-lite-latest", False),
            ("gemini-3.1-flash-lite", False),
            ("gemini-3.5-flash", False),
        ]

        last_err = None
        for model_name, supports_thinking_zero in models_to_try:
            try:
                cfg_kwargs = {
                    "response_mime_type": "application/json",
                    "response_schema": GroundedOutfitResponse,
                    "temperature": 0.0,
                }
                if supports_thinking_zero:
                    cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

                response = client.models.generate_content(
                    model=model_name,
                    contents=[prompt, img_part],
                    config=types.GenerateContentConfig(**cfg_kwargs),
                )
                parsed = GroundedOutfitResponse.model_validate_json(response.text)

                # Enforce strict consistency on person_count flags
                if parsed.person_count == 0:
                    parsed.has_no_person = True
                    parsed.is_group_photo = False
                    parsed.items = []
                    if not parsed.rejection_reason:
                        parsed.rejection_reason = (
                            "Can't detect such photo: No person is present in this photo. "
                            "Please upload a photo with a single person."
                        )
                elif parsed.person_count > 1 or parsed.is_group_photo:
                    parsed.is_group_photo = True
                    parsed.has_no_person = False
                    parsed.items = []
                    if not parsed.rejection_reason:
                        parsed.rejection_reason = (
                            "Can't detect such photo: Group photos are not allowed. "
                            "Please upload a photo with a single person."
                        )

                logger.info(
                    f"[GroundedDetector] Success with {model_name} "
                    f"(persons={parsed.person_count}, group={parsed.is_group_photo}, no_person={parsed.has_no_person}, items={len(parsed.items)})"
                )
                return parsed.model_dump()
            except Exception as ex:
                logger.warning(f"[GroundedDetector] Model {model_name} failed: {ex}")
                last_err = ex

        if last_err:
            raise RuntimeError(f"Gemini Vision API failed across all models: {last_err}")
        return {
            "person_count": 0,
            "is_group_photo": False,
            "has_no_person": True,
            "rejection_reason": "Can't detect such photo: Unable to detect a single person in this photo.",
            "items": []
        }


def generate_garment_masks(img_rgb: np.ndarray, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts Gemini bounding boxes into precise garment masks using GrabCut,
    applies layer subtraction (outer minus inner, inner minus outer),
    and removes skin pixels using anatomical skin detection.
    """
    h, w = img_rgb.shape[:2]
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    raw_masks = []
    processed_items = []

    for item in items:
        box_raw = item["box_2d"]
        box_norm, box_px = normalize_box(box_raw, h, w)
        ymin, xmin, ymax, xmax = box_px

        rect_w = max(1, xmax - xmin)
        rect_h = max(1, ymax - ymin)
        rect = (xmin, ymin, rect_w, rect_h)

        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        gc_mask = np.zeros((h, w), np.uint8)

        try:
            cv2.grabCut(img_bgr, gc_mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
            mask = np.where((gc_mask == 2) | (gc_mask == 0), 0, 255).astype(np.uint8)
        except Exception:
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[ymin:ymax, xmin:xmax] = 255

        bounded_mask = np.zeros((h, w), dtype=np.uint8)
        bounded_mask[ymin:ymax, xmin:xmax] = mask[ymin:ymax, xmin:xmax]

        # Fallback if GrabCut collapsed on small accessories (e.g. watch, belt, sunglasses)
        if np.count_nonzero(bounded_mask) < 25:
            bounded_mask[ymin:ymax, xmin:xmax] = 255

        raw_masks.append(bounded_mask)
        processed_items.append((item, box_norm, box_px))

    # Layer resolution: outer garments subtract inner tops, inner tops subtract outer jackets
    resolved_masks = []
    for i, (item_i, norm_i, px_i) in enumerate(processed_items):
        mask_i = raw_masks[i].copy()
        layer_i = item_i.get("layer", "base")
        cat_i = item_i.get("category", "")

        for j, (item_j, norm_j, px_j) in enumerate(processed_items):
            if i == j:
                continue
            layer_j = item_j.get("layer", "base")
            cat_j = item_j.get("category", "")

            # If i is outer and j is inner top, outer should not claim inner top center
            if layer_i == "outer" and layer_j == "inner" and cat_i in ["overtop", "top"] and cat_j == "top":
                mask_i = cv2.bitwise_and(mask_i, cv2.bitwise_not(raw_masks[j]))

            # If i is inner and j is outer, inner top should not overlap the outer blazer lapels
            if layer_i == "inner" and layer_j == "outer" and cat_i == "top" and cat_j in ["overtop", "top"]:
                overlap = cv2.bitwise_and(mask_i, raw_masks[j])
                if np.count_nonzero(overlap) > 0:
                    eroded_outer = cv2.erode(raw_masks[j], np.ones((7, 7), np.uint8))
                    mask_i = cv2.bitwise_and(mask_i, cv2.bitwise_not(eroded_outer))

        resolved_masks.append(mask_i)

    # Skin mask for non-footwear/non-accessory garments
    skin_mask = extract_skin_mask(img_rgb)

    results = []
    for i, (item, box_norm, box_px) in enumerate(processed_items):
        mask = resolved_masks[i]
        if skin_mask is not None and item.get("category") not in ["footwear", "accessory"]:
            non_skin_mask = cv2.bitwise_and(mask, cv2.bitwise_not(skin_mask))
            if np.count_nonzero(non_skin_mask) > 100:
                mask = non_skin_mask

        ymin, xmin, ymax, xmax = box_px
        crop_rgb = img_rgb[ymin:ymax, xmin:xmax]
        crop_mask = mask[ymin:ymax, xmin:xmax]

        crop_rgba = np.zeros((crop_rgb.shape[0], crop_rgb.shape[1], 4), dtype=np.uint8)
        crop_rgba[:, :, :3] = crop_rgb
        crop_rgba[:, :, 3] = crop_mask

        ys, xs = np.where(crop_mask > 0)
        if len(ys) > 0 and len(xs) > 0:
            center_y = float(ymin + np.mean(ys))
            center_x = float(xmin + np.mean(xs))
        else:
            center_y = float(ymin + (ymax - ymin) / 2.0)
            center_x = float(xmin + (xmax - xmin) / 2.0)

        area = int(np.count_nonzero(crop_mask))
        results.append({
            "item_data": item,
            "crop_rgb": crop_rgb,
            "crop_rgba": crop_rgba,
            "mask": mask,
            "crop_mask": crop_mask,
            "bbox": box_px,
            "bbox_norm": box_norm,
            "mask_center": [center_y, center_x],
            "area": area
        })

    return results
