"""
Color & Palette Extraction Engine in CIELAB Color Space.
Operates strictly on eroded mask pixels with hue-aware matching.
No crop-level gray-world distortion.
Uses K-Means clustering and Delta-E perceptual distance for:
- Deterministic shade distinction (Pure White vs Off-White vs Cream, Navy vs Royal Blue vs Denim)
- Multicolor / Patterned item detection
- Hex palette with cluster coverage percentages.
"""

import cv2
import numpy as np

REFERENCE_SHADES = [
    # Whites & Neutrals
    {"name": "pure_white", "family": "neutral", "primary": "white", "lab": (98.0, 0.0, 0.0), "hex": "#FFFFFF"},
    {"name": "off_white", "family": "neutral", "primary": "white", "lab": (93.0, 0.0, 4.0), "hex": "#FAF9F6"},
    {"name": "cream", "family": "warm", "primary": "cream", "lab": (91.0, -1.0, 10.0), "hex": "#FFFDD0"},
    {"name": "ivory", "family": "warm", "primary": "white", "lab": (94.0, -2.0, 8.0), "hex": "#FFFFF0"},
    {"name": "light_grey", "family": "neutral", "primary": "grey", "lab": (76.0, 0.0, 0.0), "hex": "#C0C0C0"},
    {"name": "grey", "family": "neutral", "primary": "grey", "lab": (52.0, 0.0, 0.0), "hex": "#808080"},
    {"name": "charcoal_grey", "family": "neutral", "primary": "grey", "lab": (28.0, 0.0, 0.0), "hex": "#36454F"},
    {"name": "black", "family": "neutral", "primary": "black", "lab": (8.0, 0.0, 0.0), "hex": "#000000"},

    # Blues & Denim
    {"name": "dark_navy", "family": "blue", "primary": "navy", "lab": (16.0, 2.0, -15.0), "hex": "#0A192F"},
    {"name": "navy", "family": "blue", "primary": "navy", "lab": (22.0, 4.0, -24.0), "hex": "#000080"},
    {"name": "indigo_denim", "family": "blue", "primary": "blue", "lab": (30.0, -1.0, -22.0), "hex": "#1F3A52"},
    {"name": "faded_denim", "family": "blue", "primary": "blue", "lab": (56.0, -2.0, -15.0), "hex": "#5B7C99"},
    {"name": "blue", "family": "blue", "primary": "blue", "lab": (42.0, 0.0, -32.0), "hex": "#2B60DE"},
    {"name": "royal_blue", "family": "blue", "primary": "blue", "lab": (46.0, 20.0, -60.0), "hex": "#4169E1"},
    {"name": "sky_blue", "family": "blue", "primary": "blue", "lab": (72.0, -8.0, -24.0), "hex": "#87CEEB"},
    {"name": "teal", "family": "cool", "primary": "teal", "lab": (42.0, -28.0, -12.0), "hex": "#008080"},

    # Greens & Earthy
    {"name": "olive_green", "family": "olive_green", "primary": "olive_green", "lab": (42.0, -4.0, 16.0), "hex": "#556B2F"},
    {"name": "khaki", "family": "earthy", "primary": "khaki", "lab": (62.0, -2.0, 18.0), "hex": "#C3B091"},
    {"name": "emerald_green", "family": "cool", "primary": "green", "lab": (44.0, -42.0, 18.0), "hex": "#50C878"},
    {"name": "forest_green", "family": "cool", "primary": "green", "lab": (26.0, -22.0, 12.0), "hex": "#228B22"},
    {"name": "mint", "family": "cool", "primary": "mint", "lab": (80.0, -24.0, 8.0), "hex": "#98FF98"},

    # Warm & Earthy
    {"name": "mustard", "family": "warm", "primary": "mustard", "lab": (66.0, 10.0, 60.0), "hex": "#FFDB58"},
    {"name": "yellow", "family": "warm", "primary": "yellow", "lab": (86.0, -4.0, 82.0), "hex": "#FFFF00"},
    {"name": "orange", "family": "warm", "primary": "orange", "lab": (60.0, 48.0, 62.0), "hex": "#FFA500"},
    {"name": "rust", "family": "earthy", "primary": "rust", "lab": (48.0, 32.0, 34.0), "hex": "#B7410E"},
    {"name": "beige", "family": "earthy", "primary": "beige", "lab": (82.0, 2.0, 16.0), "hex": "#F5F5DC"},
    {"name": "tan", "family": "earthy", "primary": "tan", "lab": (66.0, 8.0, 24.0), "hex": "#D2B48C"},
    {"name": "camel", "family": "earthy", "primary": "brown", "lab": (60.0, 10.0, 30.0), "hex": "#C19A6B"},
    {"name": "brown", "family": "earthy", "primary": "brown", "lab": (34.0, 14.0, 20.0), "hex": "#8B4513"},

    # Reds, Pinks, Purples
    {"name": "red", "family": "warm", "primary": "red", "lab": (48.0, 68.0, 48.0), "hex": "#FF0000"},
    {"name": "maroon", "family": "warm", "primary": "maroon", "lab": (24.0, 35.0, 15.0), "hex": "#800000"},
    {"name": "burgundy", "family": "warm", "primary": "burgundy", "lab": (22.0, 28.0, 6.0), "hex": "#800020"},
    {"name": "pink", "family": "warm", "primary": "pink", "lab": (66.0, 38.0, 2.0), "hex": "#FFC0CB"},
    {"name": "baby_pink", "family": "warm", "primary": "pink", "lab": (82.0, 18.0, 4.0), "hex": "#F4C2C2"},
    {"name": "hot_pink", "family": "warm", "primary": "pink", "lab": (52.0, 68.0, -4.0), "hex": "#FF69B4"},
    {"name": "purple", "family": "cool", "primary": "purple", "lab": (34.0, 42.0, -38.0), "hex": "#800080"},
    {"name": "lavender", "family": "cool", "primary": "lavender", "lab": (70.0, 18.0, -22.0), "hex": "#E6E6FA"},
]

def rgb_to_cielab(rgb_pixels):
    """Converts (N, 3) RGB [0..255] to standard CIELAB [L*:0..100, a*:-128..127, b*:-128..127]."""
    if len(rgb_pixels) == 0:
        return np.empty((0, 3), dtype=np.float32)
    reshaped = rgb_pixels.reshape(-1, 1, 3)
    lab_cv = cv2.cvtColor(reshaped, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    l_star = lab_cv[:, 0] * (100.0 / 255.0)
    a_star = lab_cv[:, 1] - 128.0
    b_star = lab_cv[:, 2] - 128.0
    return np.column_stack([l_star, a_star, b_star])

def lab_to_hex(lab_point):
    """Converts a standard CIELAB point back to hex color."""
    l_cv = int(np.clip(lab_point[0] * (255.0 / 100.0), 0, 255))
    a_cv = int(np.clip(lab_point[1] + 128.0, 0, 255))
    b_cv = int(np.clip(lab_point[2] + 128.0, 0, 255))
    lab_pixel = np.uint8([[[l_cv, a_cv, b_cv]]])
    rgb_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2RGB)[0, 0]
    return f"#{rgb_pixel[0]:02X}{rgb_pixel[1]:02X}{rgb_pixel[2]:02X}"

def delta_e(lab1, lab2):
    """Euclidean CIELAB color difference (CIE76)."""
    return float(np.sqrt(
        (lab1[0] - lab2[0]) ** 2 +
        (lab1[1] - lab2[1]) ** 2 +
        (lab1[2] - lab2[2]) ** 2
    ))

def match_nearest_shade(lab_point, subcategory=None):
    """
    Matches a CIELAB point to the closest named fashion shade with hue-awareness.
    Decides denim ONLY if subcategory is denim-friendly.
    """
    L, a, b = float(lab_point[0]), float(lab_point[1]), float(lab_point[2])
    chroma = float(np.sqrt(a ** 2 + b ** 2))

    # 1. High-Luminance Whites / Off-Whites / Creams
    if L >= 72.0 and chroma < 9.5:
        if b < 2.0 and abs(a) < 2.5 and L >= 82.0:
            return {"name": "pure_white", "family": "neutral", "primary": "white", "hex": "#FFFFFF"}
        elif b < 6.0 and abs(a) < 9.0:
            return {"name": "off_white", "family": "neutral", "primary": "white", "hex": "#FAF9F6"}
        elif b >= 6.0:
            return {"name": "cream", "family": "warm", "primary": "cream", "hex": "#FFFDD0"}

    # 2. Strong Blue & Denim (deep negative b*)
    if b <= -16.0 and a < 20.0:
        if L < 18.0:
            return {"name": "navy", "family": "blue", "primary": "navy", "hex": "#000080"}
        elif 18.0 <= L <= 45.0:
            if subcategory is None or subcategory in ["jeans", "denim_jacket", "denim_skirt", "denim_shorts"]:
                return {"name": "indigo_denim", "family": "blue", "primary": "blue", "hex": "#1F3A52"}
            return {"name": "blue", "family": "blue", "primary": "blue", "hex": "#2B60DE"}
        elif 45.0 < L <= 72.0:
            if subcategory is None or subcategory in ["jeans", "denim_jacket", "denim_skirt", "denim_shorts"]:
                return {"name": "faded_denim", "family": "blue", "primary": "blue", "hex": "#5B7C99"}
            return {"name": "blue", "family": "blue", "primary": "blue", "hex": "#2B60DE"}

    # 3. Dark Navy (subtle cool tint, low chroma)
    if -16.0 < b <= -1.5 and -10.0 <= a <= 15.0 and L <= 36.0:
        return {"name": "dark_navy", "family": "blue", "primary": "navy", "hex": "#0A192F"}

    # 4. Ultra-Dark Neutral (Black)
    if L < 15.0 and chroma < 4.5:
        return {"name": "black", "family": "neutral", "primary": "black", "hex": "#000000"}

    # 5. Moderate Blue & Denim (negative b*)
    if b <= -8.0 and a < 16.0:
        if subcategory in ["jeans", "denim_jacket", "denim_skirt", "denim_shorts"]:
            return {"name": "faded_denim", "family": "blue", "primary": "blue", "hex": "#5B7C99"}
        if L < 35.0:
            return {"name": "dark_navy", "family": "blue", "primary": "navy", "hex": "#0A192F"}
        elif L < 60.0:
            return {"name": "blue", "family": "blue", "primary": "blue", "hex": "#2B60DE"}
        else:
            return {"name": "sky_blue", "family": "blue", "primary": "blue", "hex": "#87CEEB"}
    elif b <= -2.0 and a < 16.0:
        if subcategory in ["jeans", "denim_jacket", "denim_skirt", "denim_shorts"]:
            return {"name": "faded_denim", "family": "blue", "primary": "blue", "hex": "#5B7C99"}
        if L < 36.0:
            return {"name": "dark_navy", "family": "blue", "primary": "navy", "hex": "#0A192F"}

    # 5. Olive Green / Khaki (positive b, negative or near-zero a, moderate L)
    if b >= 4.0 and -16.0 <= a <= 6.0 and 24.0 <= L <= 75.0:
        if L < 54.0:
            return {"name": "olive_green", "family": "olive_green", "primary": "olive_green", "hex": "#556B2F"}
        else:
            return {"name": "khaki", "family": "earthy", "primary": "khaki", "hex": "#C3B091"}

    # 6. Fallback: Minimum Delta-E nearest neighbor match across reference shades
    best_dist = float("inf")
    best_match = REFERENCE_SHADES[0]
    for ref in REFERENCE_SHADES:
        d = delta_e(lab_point, ref["lab"])
        if d < best_dist:
            best_dist = d
            best_match = ref
    return best_match

def extract_dominant_colors(image_rgb_or_bgr, mask=None, subcategory=None, k=4, delta_e_thresh=8.0, is_bgr=False):
    """
    Extracts deterministic color information from eroded mask pixels.
    No crop-level gray-world distortion.
    Filters out skin and extreme shadow pixels.
    """
    if is_bgr:
        img_rgb = cv2.cvtColor(image_rgb_or_bgr, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = image_rgb_or_bgr

    h, w = img_rgb.shape[:2]

    if mask is not None:
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Erode mask by a few pixels to eliminate boundary/edge bleed
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        eroded = cv2.erode(mask, kernel, iterations=1)
        if np.sum(eroded > 0) > 50:
            active_mask = (eroded > 0)
        else:
            active_mask = (mask > 0)
    else:
        # Center crop if no mask provided
        active_mask = np.zeros((h, w), dtype=bool)
        active_mask[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)] = True

    pixels = img_rgb[active_mask]
    if len(pixels) < 40:
        return {
            "primary": "white",
            "shade": "pure_white",
            "secondary": None,
            "family": "neutral",
            "is_multicolor": False,
            "palette": [{"shade": "pure_white", "hex": "#FFFFFF", "percent": 100.0}]
        }

    # Drop human skin pixels (illuminated skin and neck shadow folds)
    R = pixels[:, 0].astype(int)
    G = pixels[:, 1].astype(int)
    B = pixels[:, 2].astype(int)
    skin_bright = (R > 75) & (G > 35) & (B > 25) & (R > G + 12) & (R > B + 20) & (G >= B - 5)
    skin_shadow = (R < 95) & (G < 75) & (B < 65) & (R > G + 8) & (R > B + 14)
    skin_mask = skin_bright | skin_shadow
    non_skin = pixels[~skin_mask]
    if len(non_skin) >= 40:
        pixels = non_skin

    # Drop extreme pitch black shadow pixels if non-shadow pixels are substantial
    shadow_mask = (pixels[:, 0] < 12) & (pixels[:, 1] < 12) & (pixels[:, 2] < 12)
    non_shadow = pixels[~shadow_mask]
    if len(non_shadow) >= 40:
        pixels = non_shadow

    # Drop white studio background bleed specifically for jeans/denim
    if subcategory in ["jeans", "denim_jacket"]:
        is_studio_white = (pixels[:, 0] > 195) & (pixels[:, 1] > 195) & (pixels[:, 2] > 195)
        non_white = pixels[~is_studio_white]
        if len(non_white) >= 50:
            pixels = non_white

    # Convert directly to CIELAB (NO Gray-World!)
    lab_pixels = rgb_to_cielab(pixels)

    # Subsample if large for performance (deterministic)
    np.random.seed(42)
    cv2.setRNGSeed(42)
    if len(lab_pixels) > 6000:
        idx = np.random.choice(len(lab_pixels), 6000, replace=False)
        lab_pixels = lab_pixels[idx]

    # K-Means in CIELAB
    unique_pixels = np.unique(lab_pixels, axis=0)
    k_actual = min(k, len(unique_pixels))
    if k_actual < 2:
        shade_info = match_nearest_shade(np.mean(lab_pixels, axis=0), subcategory=subcategory)
        return {
            "primary": shade_info["primary"],
            "shade": shade_info["name"],
            "secondary": None,
            "family": shade_info["family"],
            "is_multicolor": False,
            "palette": [{"shade": shade_info["name"], "hex": lab_to_hex(np.mean(lab_pixels, axis=0)), "percent": 100.0}]
        }

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    data = lab_pixels.astype(np.float32)
    _, labels_arr, centers = cv2.kmeans(data, k_actual, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    labels = labels_arr.flatten()

    counts = np.bincount(labels, minlength=k_actual)
    percentages = (counts / len(lab_pixels)) * 100.0

    clusters = [{"lab": centers[i], "percent": float(percentages[i])} for i in range(k_actual)]
    clusters.sort(key=lambda x: x["percent"], reverse=True)

    # Merge near-duplicate clusters based on Delta-E
    merged = []
    for c in clusters:
        matched = False
        for m in merged:
            if delta_e(c["lab"], m["lab"]) < delta_e_thresh:
                total = m["percent"] + c["percent"]
                if total > 0:
                    m["lab"] = (m["lab"] * m["percent"] + c["lab"] * c["percent"]) / total
                m["percent"] = total
                matched = True
                break
        if not matched:
            merged.append(c)

    merged.sort(key=lambda x: x["percent"], reverse=True)

    palette = []
    for c in merged:
        shade_info = match_nearest_shade(c["lab"], subcategory=subcategory)
        hex_code = lab_to_hex(c["lab"])
        palette.append({
            "shade": shade_info["name"],
            "primary": shade_info["primary"],
            "family": shade_info["family"],
            "hex": hex_code,
            "percent": round(c["percent"], 1)
        })

    # Aggregate by shade to avoid K-Means splitting of the dominant color
    shade_totals = {}
    for p in palette:
        sh = p["shade"]
        shade_totals[sh] = shade_totals.get(sh, 0.0) + p["percent"]

    best_shade = max(shade_totals.items(), key=lambda x: x[1])[0]
    primary_cluster = next(p for p in palette if p["shade"] == best_shade)
    
    other_clusters = [p for p in palette if p["shade"] != best_shade]
    secondary_cluster = other_clusters[0] if other_clusters and other_clusters[0]["percent"] >= 12.0 else None

    # Detect Multicolor
    is_multicolor = False
    if secondary_cluster and secondary_cluster["percent"] >= 22.0:
        d = delta_e(merged[0]["lab"], merged[1]["lab"])
        if d >= 16.0:
            is_multicolor = True

    return {
        "primary": primary_cluster["primary"],
        "shade": primary_cluster["shade"],
        "secondary": secondary_cluster["shade"] if secondary_cluster else None,
        "family": primary_cluster["family"],
        "is_multicolor": is_multicolor,
        "palette": palette
    }
