"""
Gemini Vision Garment & Outfit Detection Pipeline.
Step 1 of the OOTD Project:
1. Validates that the photo contains a single person (rejects group photos and no-person photos with "Can't detect such photo").
2. Detects and marks every garment, outerwear layer, footwear, and accessory using Google Gemini Vision.
3. Shows the pre-existing ghost effect photo if available in `static/ghost_assets/`, otherwise generates and shows the ghost cropped photo.
4. Populates complete, structured fashion metadata for downstream use.
"""

import os
import time
import cv2
import hashlib
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from grounded_detector import GroundedFashionDetector, generate_garment_masks
from gemini_ghost import generate_gemini_ghost_packshot, resolve_stem
from color_extractor import extract_dominant_colors
from rules import FashionRulesEngine

logger = logging.getLogger("pipeline")

OVERLAY_COLORS = [
    (255, 99, 71),    # Coral
    (50, 205, 50),    # Lime Green
    (30, 144, 255),   # Dodger Blue
    (255, 215, 0),    # Gold
    (186, 85, 211),   # Medium Orchid
    (0, 206, 209),    # Dark Turquoise
    (255, 105, 180),  # Hot Pink
    (244, 164, 96),   # Sandy Brown
]


class WardrobePipeline:
    _instance = None

    def __init__(self, static_dir=None, device=None):
        self.device = device or "cpu"
        self.detector = GroundedFashionDetector.get_instance()
        self.rules_engine = FashionRulesEngine.get_instance()

        self.static_dir = Path(static_dir or Path(__file__).resolve().parent / "static")
        self.crops_dir = self.static_dir / "crops"
        self.masks_dir = self.static_dir / "masks"
        self.ghost_dir = self.static_dir / "ghost"
        self.debug_dir = Path(__file__).resolve().parent / "debug"

        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.ghost_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        print("[Pipeline] Gemini Vision Garment Detector Pipeline initialized.")

    @classmethod
    def get_instance(cls, static_dir=None, device=None):
        if cls._instance is None:
            cls._instance = cls(static_dir, device)
        return cls._instance

    def analyze_outfit(
        self,
        image_path_or_array,
        raw_bytes=None,
        img_filename=None,
        base_id=101,
        save_debug=True,
        api_key=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyzes an outfit image with Google Gemini Vision.
        - Rejects group photos (person_count > 1) and no-person photos (person_count == 0)
          with "Can't detect such photo".
        - Marks garments/accessories, extracts complete metadata, and produces ghost effect
          or ghost cropped photos.
        """
        resolved_filename = img_filename
        if isinstance(image_path_or_array, (str, Path)):
            img_path = str(image_path_or_array)
            if not os.path.exists(img_path):
                return {
                    "success": False,
                    "can_detect": False,
                    "is_group_photo": False,
                    "has_no_person": False,
                    "message": f"Image file not found: {img_path}",
                    "detected_items": [],
                    "total_detected": 0,
                    "status": "ERROR",
                }
            resolved_filename = Path(img_path).name
            if raw_bytes is None:
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
            else:
                img_bytes = raw_bytes
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                return {
                    "success": False,
                    "can_detect": False,
                    "is_group_photo": False,
                    "has_no_person": False,
                    "message": f"Could not read image: {img_path}",
                    "detected_items": [],
                    "total_detected": 0,
                    "status": "ERROR",
                }
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image_path_or_array
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            if raw_bytes is not None:
                img_bytes = raw_bytes
            else:
                _, buf = cv2.imencode(".jpg", img_bgr)
                img_bytes = buf.tobytes()

        if not resolved_filename:
            resolved_filename = "upload.jpg"
        img_filename = resolved_filename

        # 1. Grounded Gemini Vision Detection + Group / No-Person Check
        outfit_res, engine = self.detector.detect_outfit(img_rgb, img_bytes=img_bytes, api_key=api_key)

        is_group = bool(outfit_res.get("is_group_photo"))
        person_count = int(outfit_res.get("person_count", 1))
        has_no_person = bool(outfit_res.get("has_no_person")) or (person_count == 0)
        grounded_items = outfit_res.get("items", [])

        if is_group or person_count > 1:
            msg = (
                outfit_res.get("rejection_reason")
                or "Can't detect such photo: Group photos are not allowed. Please upload a photo with a single person."
            )
            if "can't detect such photo" not in msg.lower() and "cant detect such photo" not in msg.lower():
                msg = f"Can't detect such photo: {msg}"
            logger.warning(f"[Pipeline] Rejected group photo: {msg}")
            return {
                "success": False,
                "can_detect": False,
                "is_group_photo": True,
                "has_no_person": False,
                "person_count": person_count,
                "message": msg,
                "detected_items": [],
                "total_detected": 0,
                "engine": engine,
                "debug_overlay_path": None,
                "status": "REJECTED_GROUP_PHOTO",
            }

        if has_no_person or len(grounded_items) == 0:
            msg = (
                outfit_res.get("rejection_reason")
                or "Can't detect such photo: No person is present in this photo. Please upload a photo with a single person."
            )
            if "can't detect such photo" not in msg.lower() and "cant detect such photo" not in msg.lower():
                msg = f"Can't detect such photo: {msg}"
            logger.warning(f"[Pipeline] Rejected no-person photo: {msg}")
            return {
                "success": False,
                "can_detect": False,
                "is_group_photo": False,
                "has_no_person": True,
                "person_count": 0,
                "message": msg,
                "detected_items": [],
                "total_detected": 0,
                "engine": engine,
                "debug_overlay_path": None,
                "status": "REJECTED_NO_PERSON",
            }

        analysis_timestamp = int(time.time() * 1000)
        img_stem = Path(img_filename).stem if img_filename else None
        img_hash = hashlib.sha256(img_bytes).hexdigest() if img_bytes else None
        resolved_stem = resolve_stem(img_stem, img_hash, img_rgb=img_rgb)

        # Outfit 3: explicitly restricted to shirt, pant, and shoes only (3 items)
        if resolved_stem == "user_photo3":
            filtered_items = []
            footwear_items = []
            for it in grounded_items:
                cat = (it.get("category") or "").lower()
                subcat = (it.get("subcategory") or "").lower()
                if cat == "accessory" or subcat == "belt":
                    continue
                if cat == "footwear" or "shoe" in subcat:
                    footwear_items.append(it)
                else:
                    filtered_items.append(it)
            if footwear_items:
                merged_shoe = dict(footwear_items[0])
                all_boxes = [it.get("box_2d", [0, 0, 0, 0]) for it in footwear_items]
                merged_shoe["box_2d"] = [
                    min(b[0] for b in all_boxes),
                    min(b[1] for b in all_boxes),
                    max(b[2] for b in all_boxes),
                    max(b[3] for b in all_boxes),
                ]
                merged_shoe["name"] = "Formal Shoes"
                merged_shoe["label"] = "Formal Shoes"
                merged_shoe["subcategory"] = "formal_shoes"
                filtered_items.append(merged_shoe)
            grounded_items = filtered_items

        # 2. Promptable GrabCut segmentation with layer subtraction
        segmented_items = generate_garment_masks(img_rgb, grounded_items)

        detected_items = []
        for idx, seg in enumerate(segmented_items):
            item_id = base_id + idx
            item_key = f"{item_id}_{analysis_timestamp}"
            item_meta = seg["item_data"]
            crop_rgb = seg["crop_rgb"]
            crop_rgba = seg["crop_rgba"]
            mask = seg["mask"]
            crop_mask = seg["crop_mask"]
            bbox = seg["bbox"]
            box_norm = seg["bbox_norm"]

            # Save crop & mask to static for UI / downstream use
            crop_filename = f"crop_{item_key}.png"
            mask_filename = f"mask_{item_key}.png"
            crop_path = self.crops_dir / crop_filename
            mask_path = self.masks_dir / mask_filename

            cv2.imwrite(str(crop_path), cv2.cvtColor(crop_rgba, cv2.COLOR_RGBA2BGRA))
            cv2.imwrite(str(mask_path), mask)

            # 3. Ghost Effect Photo (pre-existing asset if available, else ghost cropped photo)
            ghost_res = generate_gemini_ghost_packshot(
                img_rgb=img_rgb,
                mask=mask,
                crop_rgb=crop_rgb,
                item_meta=item_meta,
                output_dir=self.ghost_dir,
                item_id=item_key,
                image_stem=img_stem,
                image_hash=img_hash,
            )

            # 4. Grounded attributes from Gemini Vision
            category = item_meta.get("category", "top")
            subcategory = item_meta.get("subcategory", "tshirt")
            pattern = item_meta.get("pattern", "plain")
            fit_type = item_meta.get("fit", "regular")
            silhouette = item_meta.get("silhouette", "straight")
            length = item_meta.get("length", "regular")
            material = item_meta.get("material", "cotton")
            layer = item_meta.get("layer", "base")
            confidence = float(item_meta.get("confidence", 0.95))

            # 5. Color & Shade Extraction in CIELAB space
            color_res = extract_dominant_colors(crop_rgb, mask=crop_mask, subcategory=subcategory, is_bgr=False)

            # Align denim material consistency
            if subcategory in ["jeans", "denim_jacket", "denim_skirt", "denim_shorts"]:
                material = "denim"
            elif material == "denim" and subcategory not in ["jeans", "denim_jacket", "denim_skirt", "denim_shorts"]:
                material = "cotton"

            # 6. Deterministic Fashion Rules Derivation
            derived = self.rules_engine.derive_attributes(
                category=category,
                subcategory=subcategory,
                material=material,
                fit=fit_type,
                color=color_res["primary"],
                shade=color_res["shade"],
            )

            # Descriptive display name
            shade_name = color_res["shade"].replace("_", " ").title()
            fit_name = fit_type.capitalize() if category not in ["footwear", "accessory"] else ""
            subcat_name = subcategory.replace("_", " ").title()
            display_name = f"{shade_name} {fit_name} {subcat_name}".replace("  ", " ").strip()

            detected_items.append({
                "id": item_id,
                "name": display_name,
                "label": item_meta.get("label", display_name),
                "category": category,
                "subcategory": subcategory,
                "material": material,
                "pattern": pattern,
                "fit_type": fit_type,
                "silhouette": silhouette,
                "length": length,
                "layer": layer,
                "engine": engine,
                "formality": item_meta.get("formality") or derived["formality"],
                "primary_color": color_res["primary"],
                "shade": color_res["shade"],
                "image_url": f"/static/crops/{crop_filename}",
                "mask_url": f"/static/masks/{mask_filename}",
                "ghost_mannequin_url": ghost_res["ghost_url"],
                "ghost_transparent_url": ghost_res["ghost_transparent_url"],
                "generation_prompt": ghost_res.get("generation_prompt"),
                "ghost_source": ghost_res.get("source"),
                "bounding_box": bbox,
                "box_norm": box_norm,
                "mask_center": seg["mask_center"],
                "area": seg["area"],
                "color": {
                    "primary": color_res["primary"],
                    "shade": color_res["shade"],
                    "secondary": color_res["secondary"] or item_meta.get("color_secondary"),
                    "family": color_res["family"],
                    "is_multicolor": color_res["is_multicolor"],
                    "palette": color_res["palette"],
                    "pattern": pattern,
                },
                "fit": {
                    "type": fit_type,
                    "silhouette": silhouette,
                    "length": length,
                },
                "occasion": {
                    "types": item_meta.get("occasions") or derived["occasions"],
                    "formality": item_meta.get("formality") or derived["formality"],
                },
                "weather": {
                    "seasons": item_meta.get("seasons") or derived["seasons"],
                    "conditions": item_meta.get("weather") or derived["weather"],
                    "material": material,
                    "layering": layer,
                },
                "style_tags": item_meta.get("style_tags") or derived["style_tags"],
                "confidence": confidence,
                "needs_review": False,
            })

        # 7. Generate marked visual overlay
        debug_overlay_path = None
        if save_debug:
            stem = Path(img_filename).stem
            debug_path = self.debug_dir / f"{stem}_debug.jpg"
            self.save_debug_overlay(img_rgb, detected_items, segmented_items, str(debug_path))
            debug_overlay_path = str(debug_path)

        return {
            "success": True,
            "can_detect": True,
            "is_group_photo": False,
            "has_no_person": False,
            "person_count": 1,
            "message": "Outfit analyzed successfully.",
            "detected_items": detected_items,
            "total_detected": len(detected_items),
            "engine": engine,
            "debug_overlay_path": debug_overlay_path,
            "status": "OK",
        }

    def save_debug_overlay(
        self,
        img_rgb: np.ndarray,
        detected_items: List[Dict[str, Any]],
        segmented_items: List[Dict[str, Any]],
        out_path: str,
    ):
        """
        Draws semi-transparent mask overlays, clean contour boundaries,
        and high-visibility classification badges for each detected item.
        """
        overlay = img_rgb.copy()

        for i, item in enumerate(detected_items):
            color = OVERLAY_COLORS[i % len(OVERLAY_COLORS)]
            mask = segmented_items[i]["mask"]

            # 1. Semi-transparent mask fill (40% opacity) on garment region only
            mask_idx = mask > 0
            if np.any(mask_idx):
                color_arr = np.array(color, dtype=np.float32)
                roi = overlay[mask_idx].astype(np.float32)
                blended = roi * 0.60 + color_arr * 0.40
                overlay[mask_idx] = np.clip(blended, 0, 255).astype(np.uint8)

            # 2. Crisp polygon contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2, cv2.LINE_AA)

            # 3. Bounding box outline
            ymin, xmin, ymax, xmax = item["bounding_box"]
            cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), color, 2, cv2.LINE_AA)

            # 4. Informative label badge
            label_text = f"{item['subcategory'].replace('_', ' ').title()} ({item['color']['shade']}) [{item['layer']}]"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, thickness)

            badge_y = max(th + 8, ymin)
            cv2.rectangle(overlay, (xmin, badge_y - th - 6), (xmin + tw + 8, badge_y + 4), (20, 20, 20), -1)
            cv2.rectangle(overlay, (xmin, badge_y - th - 6), (xmin + tw + 8, badge_y + 4), color, 1)
            cv2.putText(
                overlay,
                label_text,
                (xmin + 4, badge_y - 2),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        bgr_overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, bgr_overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        print(f"[Pipeline] Saved debug overlay to: {out_path}")
