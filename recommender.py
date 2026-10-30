"""
Smart 4-Pillar Recommendation Engine powered by Artificial Neural Networks (PyTorch ANN)
and Machine Learning (Scikit-Learn Gradient Boosting Scorer), with Sartorial & Cultural
Combination Intelligence.

Implements the 4-Pillar Algorithmic Model:
- Pillar 1: Weather & Seasonal Compatibility Filter
  * Temp > 28°C (Hot): Filter for material: cotton/linen, layering: none/base, category: top/shorts/light bottoms.
  * Temp < 15°C (Cold): Trigger outerwear pairing (category: overtop, material: wool/denim/leather/tweed/knit/silk).
  * Seasonal Alignment: Matching season adds +1.0 weight; conflicting items penalized.
- Pillar 2: Event & Formality Context Filter
  * Work / Formal Meeting: Strict filter for formality: formal/smart_casual, category: shirt + trousers/blazer/coat.
  * Casual Outing / College: Allows style_tags: streetwear, casual, category: tshirt + jeans/cargos/chinos.
  * Festival / Wedding / Ethnic: Prioritizes Kurta + Pajama/Churidar + Nehru Jacket + Mojari.
  * Plus Party, Date Night, Gym/Sports.
- Pillar 3: Color Theory, Fit & Sartorial Combination Matrix (ALWAYS ON - Main Factor)
  * Sartorial & Cultural Domain Compatibility: Understands Ethnic vs. Western Formal vs. Streetwear combinations
    (e.g., never pairs T-Shirt with Churidar/Pajama, never pairs Western Coat/Biker Jacket with Kurta,
    pairs Kurta with Pajama/Churidar + Nehru Jacket + Mojari, Shirt with Trousers/Chinos + Coat/Blazer + Oxfords/Loafers,
    and T-Shirt with Jeans/Cargos/Chinos + Biker Jacket + Sneakers).
  * Neutral Anchor Principle: Neutral colors (white, black, grey, beige, navy) pair with any color (100% Harmony).
  * Complementary Harmony: Opposite/complementary colors on color wheel (e.g., Navy/Blue + Warm Brown/Beige/Olive) receive high compatibility scores.
  * Single Color Pop Rule: 1 Vibrant item (red, yellow, pink, etc.) paired with Neutral items (black, white, grey, navy, beige).
  * Fit & Pattern Harmony: Balanced top/bottom proportions and single statement pattern rule.
- Pillar 4: Personalized Date of Birth (DOB) Lucky Color Engine
  * Planetary Calculation: Calculates Birth Number (Day digital root / mod 9) and maps current Day of Week to ruling planet to derive personalized Lucky Colors.
"""

import os
import math
import json
import itertools
import datetime
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import requests
from sklearn.ensemble import GradientBoostingRegressor

logger = logging.getLogger("recommender")
logger.setLevel(logging.INFO)

CACHE_DIR = Path(__file__).resolve().parent / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
ANN_WEIGHTS_PATH = CACHE_DIR / "outfit_recommender_ann_v2.pt"

# ---------------------------------------------------------------------------
# Sartorial & Cultural Domain Definitions
# ---------------------------------------------------------------------------
ETHNIC_TOPS = {"kurta", "kurti", "sherwani", "short_kurta", "kameez", "anarkali", "ethnic_top"}
ETHNIC_BOTTOMS = {"churidar", "pajama", "ethnic_pajama", "salwar", "dhoti", "lehenga_skirt", "sharara", "palazzo"}
ETHNIC_OUTERWEAR = {"nehru_jacket", "bundi", "ethnic_jacket", "waistcoat", "dupatta_stole", "dupatta", "shawl"}
ETHNIC_FOOTWEAR = {"mojari", "juttis", "juttis_mojaris", "kolhapuri", "ethnic_sandals"}

WESTERN_CASUAL_TOPS = {"tshirt", "polo", "hoodie", "sweatshirt", "tank_top", "crop_top"}
WESTERN_SHIRTS = {"shirt", "blouse", "top"}
WESTERN_FORMAL_BOTTOMS = {"trousers", "formal_trousers", "wide_leg_trousers", "chinos"}
WESTERN_CASUAL_BOTTOMS = {"jeans", "cargos", "cargo_pants", "joggers", "shorts", "track_pants", "sweatpants", "skirt"}
WESTERN_COATS_BLAZERS = {"coat", "overcoat", "trench_coat", "blazer", "suit_jacket"}
WESTERN_CASUAL_JACKETS = {"jacket", "biker_jacket", "leather_jacket", "bomber_jacket", "denim_jacket", "puffer_jacket", "cardigan"}
WESTERN_FORMAL_SHOES = {"oxford", "derby", "formal_shoes", "brogues", "heels"}
WESTERN_SMART_SHOES = {"loafers", "sandals", "mules", "boots"}
WESTERN_CASUAL_SHOES = {"sneakers", "sports_shoes", "chunky_sneakers", "running_shoes"}

# ---------------------------------------------------------------------------
# Color Theory Definitions (Pillar 3)
# ---------------------------------------------------------------------------
NEUTRAL_COLORS = {
    "white", "pure_white", "off_white", "cream", "ivory",
    "black", "grey", "light_grey", "charcoal_grey",
    "beige", "tan", "khaki", "navy", "dark_navy"
}

VIBRANT_POP_COLORS = {
    "red", "yellow", "mustard", "orange", "pink", "hot_pink", "baby_pink",
    "purple", "lavender", "emerald_green", "royal_blue", "teal", "burgundy", "maroon", "rust", "mint"
}

COMPLEMENTARY_PAIRS = [
    # Navy / Blue / Denim + Warm Earthy (Brown, Beige, Tan, Khaki, Olive, White, Cream, Mustard, Rust)
    ({"navy", "dark_navy", "blue", "indigo_denim", "faded_denim", "royal_blue", "sky_blue"},
     {"beige", "tan", "camel", "brown", "khaki", "olive_green", "cream", "off_white", "pure_white", "white", "mustard", "rust"}),
    # Olive / Green + Earthy / Denim / Neutrals
    ({"olive_green", "forest_green", "emerald_green", "green", "mint", "khaki"},
     {"indigo_denim", "faded_denim", "blue", "navy", "dark_navy", "beige", "tan", "cream", "white", "pure_white", "black"}),
    # Burgundy / Maroon / Red + Navy / Denim / Neutrals / Beige / Cream
    ({"burgundy", "maroon", "red", "rust"},
     {"navy", "dark_navy", "indigo_denim", "beige", "tan", "cream", "off_white", "grey", "charcoal_grey", "black", "white", "pure_white"}),
    # Mustard / Yellow + White / Cream / Navy / Blue / Beige
    ({"mustard", "yellow", "orange"},
     {"pure_white", "white", "off_white", "cream", "navy", "dark_navy", "royal_blue", "indigo_denim", "beige"}),
    # Pink / Lavender + White / Denim / Grey / Navy
    ({"pink", "baby_pink", "hot_pink", "lavender", "purple"},
     {"white", "pure_white", "off_white", "indigo_denim", "faded_denim", "navy", "light_grey", "grey", "black"}),
]

# ---------------------------------------------------------------------------
# Astrology / Planetary Calculation Definitions (Pillar 4)
# ---------------------------------------------------------------------------
BIRTH_NUMBER_PLANETS = {
    1: {
        "planet": "Sun (Surya)",
        "element": "Fire",
        "lucky_colors": ["orange", "yellow", "mustard", "red", "cream", "pure_white", "white"],
        "display_colors": [
            {"name": "Royal Gold / Mustard", "shade": "mustard", "hex": "#FFDB58"},
            {"name": "Solar Orange", "shade": "orange", "hex": "#FFA500"},
            {"name": "Warm Cream", "shade": "cream", "hex": "#FFFDD0"},
            {"name": "Pure White", "shade": "pure_white", "hex": "#FFFFFF"},
        ],
    },
    2: {
        "planet": "Moon (Chandra)",
        "element": "Water",
        "lucky_colors": ["pure_white", "off_white", "white", "cream", "light_grey", "mint", "sky_blue"],
        "display_colors": [
            {"name": "Lunar White", "shade": "pure_white", "hex": "#FFFFFF"},
            {"name": "Soft Cream", "shade": "cream", "hex": "#FFFDD0"},
            {"name": "Silver Mist", "shade": "light_grey", "hex": "#C0C0C0"},
            {"name": "Mint Green", "shade": "mint", "hex": "#98FF98"},
        ],
    },
    3: {
        "planet": "Jupiter (Brihaspati)",
        "element": "Ether",
        "lucky_colors": ["yellow", "mustard", "orange", "cream", "beige", "pink", "red"],
        "display_colors": [
            {"name": "Jupiter Yellow", "shade": "yellow", "hex": "#FFFF00"},
            {"name": "Warm Mustard", "shade": "mustard", "hex": "#FFDB58"},
            {"name": "Saffron Beige", "shade": "beige", "hex": "#F5F5DC"},
            {"name": "Rose Pink", "shade": "pink", "hex": "#FFC0CB"},
        ],
    },
    4: {
        "planet": "Uranus / Rahu",
        "element": "Air",
        "lucky_colors": ["blue", "royal_blue", "indigo_denim", "faded_denim", "sky_blue", "grey", "charcoal_grey", "khaki"],
        "display_colors": [
            {"name": "Electric Royal Blue", "shade": "royal_blue", "hex": "#4169E1"},
            {"name": "Indigo Denim", "shade": "indigo_denim", "hex": "#1F3A52"},
            {"name": "Slate Grey", "shade": "grey", "hex": "#808080"},
            {"name": "Earthy Khaki", "shade": "khaki", "hex": "#C3B091"},
        ],
    },
    5: {
        "planet": "Mercury (Budha)",
        "element": "Earth",
        "lucky_colors": ["olive_green", "emerald_green", "forest_green", "green", "mint", "teal", "pure_white", "white", "light_grey", "khaki"],
        "display_colors": [
            {"name": "Olive Green", "shade": "olive_green", "hex": "#556B2F"},
            {"name": "Emerald Green", "shade": "emerald_green", "hex": "#50C878"},
            {"name": "Teal", "shade": "teal", "hex": "#008080"},
            {"name": "Crisp White", "shade": "pure_white", "hex": "#FFFFFF"},
        ],
    },
    6: {
        "planet": "Venus (Shukra)",
        "element": "Water / Earth",
        "lucky_colors": ["pure_white", "off_white", "white", "pink", "baby_pink", "sky_blue", "blue", "lavender", "cream"],
        "display_colors": [
            {"name": "Venusian White", "shade": "pure_white", "hex": "#FFFFFF"},
            {"name": "Pastel Pink", "shade": "baby_pink", "hex": "#F4C2C2"},
            {"name": "Sky Blue", "shade": "sky_blue", "hex": "#87CEEB"},
            {"name": "Soft Lavender", "shade": "lavender", "hex": "#E6E6FA"},
        ],
    },
    7: {
        "planet": "Neptune / Ketu",
        "element": "Water",
        "lucky_colors": ["pure_white", "off_white", "white", "light_grey", "grey", "mint", "beige", "khaki", "yellow"],
        "display_colors": [
            {"name": "Ivory White", "shade": "off_white", "hex": "#FAF9F6"},
            {"name": "Mystic Grey", "shade": "light_grey", "hex": "#C0C0C0"},
            {"name": "Seafoam Mint", "shade": "mint", "hex": "#98FF98"},
            {"name": "Warm Beige", "shade": "beige", "hex": "#F5F5DC"},
        ],
    },
    8: {
        "planet": "Saturn (Shani)",
        "element": "Air / Earth",
        "lucky_colors": ["black", "dark_navy", "navy", "charcoal_grey", "indigo_denim", "blue", "purple"],
        "display_colors": [
            {"name": "Obsidian Black", "shade": "black", "hex": "#111111"},
            {"name": "Midnight Navy", "shade": "dark_navy", "hex": "#0A192F"},
            {"name": "Charcoal Grey", "shade": "charcoal_grey", "hex": "#36454F"},
            {"name": "Indigo Denim", "shade": "indigo_denim", "hex": "#1F3A52"},
        ],
    },
    9: {
        "planet": "Mars (Mangal)",
        "element": "Fire",
        "lucky_colors": ["red", "maroon", "burgundy", "rust", "pink", "hot_pink", "orange", "white", "pure_white"],
        "display_colors": [
            {"name": "Crimson Red", "shade": "red", "hex": "#FF0000"},
            {"name": "Deep Burgundy", "shade": "burgundy", "hex": "#800020"},
            {"name": "Terracotta Rust", "shade": "rust", "hex": "#B7410E"},
            {"name": "Pure White", "shade": "pure_white", "hex": "#FFFFFF"},
        ],
    },
}

WEEKDAY_RULING_PLANETS = {
    0: {
        "day_name": "Monday",
        "ruling_planet": "Moon (Chandra)",
        "lucky_colors": ["pure_white", "off_white", "white", "cream", "light_grey", "sky_blue"],
        "display_colors": [
            {"name": "Pure White", "shade": "pure_white", "hex": "#FFFFFF"},
            {"name": "Off-White", "shade": "off_white", "hex": "#FAF9F6"},
            {"name": "Silver Grey", "shade": "light_grey", "hex": "#C0C0C0"},
        ],
    },
    1: {
        "day_name": "Tuesday",
        "ruling_planet": "Mars (Mangal)",
        "lucky_colors": ["red", "maroon", "burgundy", "rust", "pink", "orange"],
        "display_colors": [
            {"name": "Mars Red", "shade": "red", "hex": "#FF0000"},
            {"name": "Burgundy", "shade": "burgundy", "hex": "#800020"},
            {"name": "Rust", "shade": "rust", "hex": "#B7410E"},
        ],
    },
    2: {
        "day_name": "Wednesday",
        "ruling_planet": "Mercury (Budha)",
        "lucky_colors": ["olive_green", "emerald_green", "forest_green", "green", "mint", "teal", "khaki"],
        "display_colors": [
            {"name": "Olive Green", "shade": "olive_green", "hex": "#556B2F"},
            {"name": "Emerald Green", "shade": "emerald_green", "hex": "#50C878"},
            {"name": "Teal", "shade": "teal", "hex": "#008080"},
        ],
    },
    3: {
        "day_name": "Thursday",
        "ruling_planet": "Jupiter (Brihaspati)",
        "lucky_colors": ["yellow", "mustard", "cream", "beige", "tan", "orange"],
        "display_colors": [
            {"name": "Mustard Gold", "shade": "mustard", "hex": "#FFDB58"},
            {"name": "Warm Beige", "shade": "beige", "hex": "#F5F5DC"},
            {"name": "Tan", "shade": "tan", "hex": "#D2B48C"},
        ],
    },
    4: {
        "day_name": "Friday",
        "ruling_planet": "Venus (Shukra)",
        "lucky_colors": ["pure_white", "off_white", "white", "pink", "baby_pink", "sky_blue", "lavender"],
        "display_colors": [
            {"name": "Diamond White", "shade": "pure_white", "hex": "#FFFFFF"},
            {"name": "Soft Pink", "shade": "baby_pink", "hex": "#F4C2C2"},
            {"name": "Sky Blue", "shade": "sky_blue", "hex": "#87CEEB"},
        ],
    },
    5: {
        "day_name": "Saturday",
        "ruling_planet": "Saturn (Shani)",
        "lucky_colors": ["black", "dark_navy", "navy", "charcoal_grey", "indigo_denim", "blue"],
        "display_colors": [
            {"name": "Saturn Black", "shade": "black", "hex": "#111111"},
            {"name": "Dark Navy", "shade": "dark_navy", "hex": "#0A192F"},
            {"name": "Indigo Denim", "shade": "indigo_denim", "hex": "#1F3A52"},
        ],
    },
    6: {
        "day_name": "Sunday",
        "ruling_planet": "Sun (Surya)",
        "lucky_colors": ["orange", "yellow", "mustard", "red", "cream", "pure_white", "white"],
        "display_colors": [
            {"name": "Solar Gold", "shade": "mustard", "hex": "#FFDB58"},
            {"name": "Bright Orange", "shade": "orange", "hex": "#FFA500"},
            {"name": "Pure White", "shade": "pure_white", "hex": "#FFFFFF"},
        ],
    },
}


def calculate_astrology_lucky_colors(dob_str: Optional[str] = "2003-05-15", target_date: Optional[datetime.date] = None) -> Dict[str, Any]:
    """
    Pillar 4: Personalized Date of Birth (DOB) Lucky Color Engine.
    - Planetary Calculation: Calculates Birth Number (Day mod 9, mapped 1..9 where 9 -> 9)
      and maps current Day of Week to its ruling planet.
    """
    if target_date is None:
        target_date = datetime.date.today()

    day = 15
    parsed_dob = dob_str or "2003-05-15"
    try:
        parts = str(parsed_dob).strip().split("-")
        if len(parts) == 3:
            day = int(parts[2])
        elif "/" in str(parsed_dob):
            sp = str(parsed_dob).strip().split("/")
            day = int(sp[0]) if int(sp[0]) <= 31 else int(sp[2])
    except Exception:
        day = 15

    day_mod_9 = day % 9
    birth_number = 9 if day_mod_9 == 0 else day_mod_9

    birth_info = BIRTH_NUMBER_PLANETS.get(birth_number, BIRTH_NUMBER_PLANETS[6])
    weekday_idx = target_date.weekday()
    weekday_info = WEEKDAY_RULING_PLANETS[weekday_idx]

    combined_shades = []
    for sh in birth_info["lucky_colors"] + weekday_info["lucky_colors"]:
        if sh not in combined_shades:
            combined_shades.append(sh)

    display_swatches = []
    seen_names = set()
    for sw in birth_info["display_colors"] + weekday_info["display_colors"]:
        if sw["shade"] not in seen_names:
            seen_names.add(sw["shade"])
            display_swatches.append(sw)

    return {
        "dob": parsed_dob,
        "birth_day": day,
        "day_mod_9": day_mod_9,
        "birth_number": birth_number,
        "birth_planet": birth_info["planet"],
        "birth_element": birth_info["element"],
        "weekday": weekday_info["day_name"],
        "weekday_planet": weekday_info["ruling_planet"],
        "lucky_shades": combined_shades,
        "birth_lucky_shades": birth_info["lucky_colors"],
        "weekday_lucky_shades": weekday_info["lucky_colors"],
        "display_swatches": display_swatches[:6],
        "summary": (
            f"Birth Day {day} (Birth # {birth_number} • {birth_info['planet']}) + "
            f"{weekday_info['day_name']} ({weekday_info['ruling_planet']})"
        ),
    }


# ---------------------------------------------------------------------------
# Live Weather Engine (Pillar 1)
# ---------------------------------------------------------------------------
CITY_COORDS = {
    "mumbai": (19.0760, 72.8777, "Mumbai"),
    "delhi": (28.6139, 77.2090, "New Delhi"),
    "new delhi": (28.6139, 77.2090, "New Delhi"),
    "bangalore": (12.9716, 77.5946, "Bengaluru"),
    "bengaluru": (12.9716, 77.5946, "Bengaluru"),
    "pune": (18.5204, 73.8567, "Pune"),
    "hyderabad": (17.3850, 78.4867, "Hyderabad"),
    "chennai": (13.0827, 80.2707, "Chennai"),
    "kolkata": (22.5726, 88.3639, "Kolkata"),
    "jaipur": (26.9124, 75.7873, "Jaipur"),
    "ahmedabad": (23.0225, 72.5714, "Ahmedabad"),
    "london": (51.5074, -0.1278, "London"),
    "new york": (40.7128, -74.0060, "New York"),
}


def fetch_today_weather(
    city: Optional[str] = "Mumbai",
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    override_temp_c: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Fetches today's live weather via Open-Meteo API and classifies it
    into Pillar 1's exact temperature regimes:
    - Temp > 28°C: Hot (filter cotton/linen, layering: none/base, top/shorts)
    - Temp < 15°C: Cold (trigger outerwear pairing: overtop, wool/denim/leather)
    - 15°C <= Temp <= 28°C: Mild / Pleasant
    """
    resolved_city = (city or "Mumbai").strip()
    lat, lon = latitude, longitude

    if lat is None or lon is None:
        key = resolved_city.lower()
        if key in CITY_COORDS:
            lat, lon, resolved_city = CITY_COORDS[key]
        else:
            try:
                geo_resp = requests.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={"name": resolved_city, "count": 1, "language": "en", "format": "json"},
                    timeout=3.0,
                )
                if geo_resp.status_code == 200:
                    results = geo_resp.json().get("results", [])
                    if results:
                        lat = float(results[0]["latitude"])
                        lon = float(results[0]["longitude"])
                        resolved_city = results[0].get("name", resolved_city)
            except Exception:
                pass

    if lat is None or lon is None:
        lat, lon = 19.0760, 72.8777

    temp_c = 29.0
    condition = "warm"
    weather_desc = "Clear & Warm"
    source = "live_open_meteo"

    if override_temp_c is not None:
        temp_c = float(override_temp_c)
        source = "manual_override"
    else:
        try:
            w_resp = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,weather_code,wind_speed_10m",
                    "timezone": "auto",
                },
                timeout=3.5,
            )
            if w_resp.status_code == 200:
                curr = w_resp.json().get("current", {})
                if "temperature_2m" in curr:
                    temp_c = float(curr["temperature_2m"])
                w_code = int(curr.get("weather_code", 0))
                if w_code in [51, 53, 55, 61, 63, 65, 80, 81, 82, 95]:
                    weather_desc = "Rainy Showers"
                elif w_code in [1, 2, 3, 45, 48]:
                    weather_desc = "Partly Cloudy"
                elif w_code in [71, 73, 75, 77]:
                    weather_desc = "Snow / Freezing"
                else:
                    weather_desc = "Sunny / Clear Sky"
            else:
                source = "seasonal_estimate"
        except Exception:
            source = "seasonal_estimate"

    if temp_c > 28.0:
        regime = "hot"
        regime_label = f"Hot ({temp_c:.1f}°C > 28°C)"
        season = "summer"
        condition = "hot"
        rule_summary = "Temp > 28°C (Hot): Prioritizing breathable cotton/linen, single base layer, light tops/bottoms."
    elif temp_c < 15.0:
        regime = "cold"
        regime_label = f"Cold ({temp_c:.1f}°C < 15°C)"
        season = "winter"
        condition = "cold"
        rule_summary = "Temp < 15°C (Cold): Triggering outerwear pairing (Overcoat/Blazer/Jacket/Nehru Jacket) + warm fabrics."
    else:
        regime = "mild"
        regime_label = f"Mild / Pleasant ({temp_c:.1f}°C)"
        season = "spring"
        condition = "mild"
        rule_summary = "15°C–28°C (Pleasant): Balanced seasonal separates & optional light layering."

    return {
        "city": resolved_city,
        "latitude": lat,
        "longitude": lon,
        "temperature_c": round(temp_c, 1),
        "regime": regime,
        "regime_label": regime_label,
        "weather_description": weather_desc,
        "condition": condition,
        "season": season,
        "rule_summary": rule_summary,
        "source": source,
    }


# ---------------------------------------------------------------------------
# Sartorial & Cultural Pairing Compatibility Engine
# ---------------------------------------------------------------------------
def evaluate_sartorial_compatibility(items: List[Dict[str, Any]]) -> Tuple[bool, float, float, List[str]]:
    """
    Evaluates whether a candidate combination of garments is sartorially & culturally coherent.
    Explicitly prevents mismatched combinations such as:
    - T-Shirt / Shirt / Polo with Churidar or Ethnic Pajama
    - Western Coat / Blazer / Biker Jacket with Kurta or Churidar/Pajama
    - Nehru Jacket with T-Shirt / Cargo Pants / Shorts / Joggers
    - Kurta with Cargo Pants / Shorts / Joggers
    - Mojari/Juttis with Western Streetwear or Coats, or Sneakers/Oxfords with Traditional Kurta+Churidar/Pajama

    Returns:
      (is_valid, top_bottom_harmony_score, layer_footwear_harmony_score, reasons)
    """
    reasons: List[str] = []
    top = next((it for it in items if it.get("category") == "top"), None)
    bottom = next((it for it in items if it.get("category") == "bottom"), None)
    overtop = next((it for it in items if it.get("category") == "overtop"), None)
    shoe = next((it for it in items if it.get("category") == "footwear"), None)

    top_sub = str(top.get("subcategory") or "").lower() if top else ""
    bot_sub = str(bottom.get("subcategory") or "").lower() if bottom else ""
    out_sub = str(overtop.get("subcategory") or "").lower() if overtop else ""
    shoe_sub = str(shoe.get("subcategory") or "").lower() if shoe else ""

    top_form = str(top.get("formality") or "casual").lower() if top else "casual"
    bot_form = str(bottom.get("formality") or "casual").lower() if bottom else "casual"
    top_pat = str(top.get("pattern") or "plain").lower() if top else "plain"
    top_fit = str(top.get("fit_type") or "regular").lower() if top else "regular"

    top_bottom_score = 0.90
    layer_shoe_score = 0.92

    # 1. TOP + BOTTOM SARTORIAL PAIRING RULES
    if top and bottom:
        # Rule 1A: Ethnic Bottoms (Churidar, Pajama, Salwar, Dhoti) ONLY pair with Ethnic Tops (Kurta, Kurti, Sherwani)
        if bot_sub in ETHNIC_BOTTOMS:
            if top_sub not in ETHNIC_TOPS:
                return False, 0.0, 0.0, ["Incompatible: Ethnic Churidar/Pajama cannot be worn with Western T-Shirt/Shirt."]
            top_bottom_score = 1.0
            reasons.append(
                f"Sartorial Pairing: Traditional {top_sub.replace('_', ' ').title()} paired authentically with Ethnic {bot_sub.replace('_', ' ').title()}."
            )

        # Rule 1B: Ethnic Tops (Kurta, Kurti, Sherwani)
        elif top_sub in ETHNIC_TOPS:
            if bot_sub in ETHNIC_BOTTOMS:
                top_bottom_score = 1.0
                reasons.append(
                    f"Sartorial Pairing: Traditional {top_sub.title()} paired with {bot_sub.title()}."
                )
            elif bot_sub == "jeans" and top_form != "formal":
                top_bottom_score = 0.85
                reasons.append("Sartorial Pairing: Smart Indo-Western Kurta paired with classic Denim Jeans.")
            else:
                # Kurta with cargos, shorts, joggers, wool dress trousers -> BLOCKED
                return False, 0.0, 0.0, [f"Incompatible: {top_sub.title()} does not pair with {bot_sub.replace('_', ' ')}."]

        # Rule 1C: Western Shirts
        elif top_sub in WESTERN_SHIRTS:
            if top_form == "formal":
                if bot_sub in {"trousers", "formal_trousers", "chinos", "wide_leg_trousers"}:
                    top_bottom_score = 1.0
                    reasons.append(f"Sartorial Pairing: Crisp Formal Shirt paired with tailored {bot_sub.replace('_', ' ').title()}.")
                elif bot_sub == "jeans":
                    top_bottom_score = 0.84
                    reasons.append("Sartorial Pairing: Smart-Casual Shirt paired with straight Denim Jeans.")
                else:
                    return False, 0.0, 0.0, ["Incompatible: Formal Dress Shirt cannot be worn with Cargos/Shorts/Joggers."]
            else:
                # Casual / Smart-Casual / Party Shirt
                if bot_sub in {"trousers", "chinos", "jeans"}:
                    top_bottom_score = 0.96
                    reasons.append(f"Sartorial Pairing: {top.get('name', 'Shirt')} paired with {bot_sub.title()}.")
                elif bot_sub in {"cargos", "shorts"} and top_form == "casual":
                    top_bottom_score = 0.86
                    reasons.append(f"Sartorial Pairing: Casual Shirt paired with relaxed {bot_sub.title()}.")
                else:
                    return False, 0.0, 0.0, ["Incompatible Shirt + Bottom combination."]

        # Rule 1D: Western Casual Tops (T-Shirt, Polo, Hoodie)
        elif top_sub in WESTERN_CASUAL_TOPS:
            if top_sub == "polo":
                if bot_sub in {"chinos", "jeans", "trousers", "shorts"}:
                    top_bottom_score = 0.98
                    reasons.append(f"Sartorial Pairing: Smart Pique Polo paired with {bot_sub.title()}.")
                elif bot_sub == "cargos":
                    top_bottom_score = 0.82
                else:
                    return False, 0.0, 0.0, ["Incompatible Polo combination."]
            else:
                # Crewneck / Oversized / Athletic T-Shirt
                top_mat = str(top.get("material") or "").lower()
                if top_mat == "polyester":
                    # Athletic Dry-Fit Gym Tee only pairs with joggers, track pants, shorts, or jeans
                    if bot_sub not in {"joggers", "track_pants", "shorts", "jeans"}:
                        return False, 0.0, 0.0, ["Incompatible: Athletic Dry-Fit Gym Tee only pairs with Gym Joggers/Shorts/Jeans."]
                if bot_sub in {"jeans", "cargos", "chinos", "joggers", "shorts"}:
                    top_bottom_score = 1.0
                    reasons.append(f"Sartorial Pairing: Casual {top_sub.upper()} paired with {bot_sub.title()}.")
                elif bot_sub in {"trousers", "formal_trousers"}:
                    # Graphic or oversized streetwear tee with formal wool dress trousers -> BLOCKED
                    if top_pat == "graphic" or top_fit == "oversized" or top_mat == "polyester":
                        return False, 0.0, 0.0, ["Incompatible: Oversized/Graphic/Athletic Tee clashes with Formal Dress Trousers."]
                    top_bottom_score = 0.82
                else:
                    return False, 0.0, 0.0, ["Incompatible T-Shirt + Bottom combination."]

    # 2. OUTERWEAR (OVERTOP) SARTORIAL COMPATIBILITY RULES
    if overtop:
        is_ethnic_base = (top_sub in ETHNIC_TOPS) or (bot_sub in ETHNIC_BOTTOMS)

        # Rule 2A: Ethnic Base (Kurta / Churidar / Pajama) ONLY pairs with Ethnic Outerwear (Nehru Jacket / Bundi / Shawl)
        if is_ethnic_base:
            if out_sub not in ETHNIC_OUTERWEAR:
                return False, 0.0, 0.0, [f"Incompatible: Cannot wear Western {out_sub.title()} over a traditional Kurta/Pajama/Churidar."]
            layer_shoe_score = 1.0
            reasons.append("Sartorial Layering: Ethnic Kurta layered with traditional Nehru Jacket (Bundi).")

        # Rule 2B: Ethnic Outerwear (Nehru Jacket / Bundi) on non-ethnic base
        elif out_sub in ETHNIC_OUTERWEAR:
            if top_sub == "shirt" and bot_sub in {"trousers", "chinos"} and top_pat == "plain":
                layer_shoe_score = 0.88
                reasons.append("Sartorial Layering: Indo-Western Bandhgala / Nehru Jacket over Crisp Shirt & Trousers.")
            else:
                return False, 0.0, 0.0, ["Incompatible: Nehru Jacket cannot be worn over T-Shirts, Streetwear, or Cargo Pants."]

        # Rule 2C: Western Coats & Blazers (Overcoat, Trench Coat, Blazer)
        elif out_sub in WESTERN_COATS_BLAZERS:
            if bot_sub in {"cargos", "cargo_pants", "shorts", "joggers", "track_pants"}:
                return False, 0.0, 0.0, [f"Incompatible: Formal {out_sub.title()} clashes with casual {bot_sub.title()}."]
            if top_sub == "tshirt" and (top_pat == "graphic" or top_fit == "oversized"):
                return False, 0.0, 0.0, [f"Incompatible: Tailored {out_sub.title()} clashes with Graphic/Oversized Streetwear Tee."]
            if top_sub in {"shirt", "polo"} and bot_sub in {"trousers", "chinos", "jeans"}:
                layer_shoe_score = 1.0
                reasons.append(f"Sartorial Layering: Structured {out_sub.title()} layered over {top_sub.title()} & {bot_sub.title()}.")
            else:
                layer_shoe_score = 0.86

        # Rule 2D: Western Casual Jackets (Leather Biker Jacket, Bomber, Puffer, Denim Jacket)
        elif out_sub in WESTERN_CASUAL_JACKETS:
            if bot_sub in {"trousers", "formal_trousers"} and bot_form == "formal":
                return False, 0.0, 0.0, ["Incompatible: Casual Biker/Streetwear Jacket clashes with Formal Dress Trousers."]
            if out_sub in {"jacket", "biker_jacket", "leather_jacket"} and bot_sub in {"joggers", "track_pants"}:
                return False, 0.0, 0.0, ["Incompatible: Leather Biker/Flight Jacket clashes with Athletic Gym Joggers."]
            if out_sub in {"puffer_jacket", "bomber_jacket"} and top_sub == "shirt" and top_form in {"formal", "semi_formal"}:
                return False, 0.0, 0.0, ["Incompatible: Athletic Bomber/Puffer Jacket clashes with Formal/Evening Dress Shirt."]
            if top_sub in {"tshirt", "polo", "shirt"} and bot_sub in {"jeans", "cargos", "chinos", "joggers"}:
                layer_shoe_score = 1.0
                reasons.append(f"Sartorial Layering: {out_sub.replace('_', ' ').title()} layered over {top_sub.upper()} & {bot_sub.title()}.")
            else:
                layer_shoe_score = 0.84

    # 3. FOOTWEAR SARTORIAL COMPATIBILITY RULES
    if shoe:
        is_ethnic_outfit = (top_sub in ETHNIC_TOPS) or (bot_sub in ETHNIC_BOTTOMS)
        is_pure_ethnic = (top_sub in ETHNIC_TOPS) and (bot_sub in ETHNIC_BOTTOMS)

        # Rule 3A: Ethnic Mojari / Juttis ONLY pair with Ethnic / Kurta outfits
        if shoe_sub in ETHNIC_FOOTWEAR:
            if not is_ethnic_outfit:
                return False, 0.0, 0.0, ["Incompatible: Traditional Ethnic Mojari/Juttis only pair with Kurtas/Ethnic wear."]
            layer_shoe_score = min(1.0, (layer_shoe_score + 1.0) / 2.0 + 0.04)
            reasons.append("Footwear Harmony: Handcrafted Ethnic Mojari/Juttis complete the traditional ensemble.")

        # Rule 3B: Pure Ethnic Outfit (Kurta + Pajama/Churidar) footwear restriction
        elif is_pure_ethnic:
            if shoe_sub in WESTERN_CASUAL_SHOES or shoe_sub in WESTERN_FORMAL_SHOES:
                return False, 0.0, 0.0, [f"Incompatible: {shoe_sub.title()} clash with traditional Kurta & {bot_sub.title()}."]
            # Loafers / Sandals acceptable with Kurta
            layer_shoe_score = min(1.0, layer_shoe_score * 0.94)
            reasons.append(f"Footwear Harmony: Classic {shoe_sub.title()} paired with traditional Kurta & {bot_sub.title()}.")

        # Rule 3C: Formal Oxford / Derby Dress Shoes
        elif shoe_sub in WESTERN_FORMAL_SHOES:
            if top_sub in WESTERN_CASUAL_TOPS or bot_sub in {"jeans", "cargos", "shorts", "joggers"} or out_sub in WESTERN_CASUAL_JACKETS:
                return False, 0.0, 0.0, ["Incompatible: Formal Oxford Dress Shoes only pair with Tailored Suit Pants/Chinos & Dress Shirts."]
            layer_shoe_score = min(1.0, (layer_shoe_score + 1.0) / 2.0 + 0.04)
            reasons.append("Footwear Harmony: Formal Cap-Toe Oxfords anchor the tailored dress silhouette.")

        # Rule 3D: Athletic Running / Sports Shoes vs Casual Sneakers
        elif shoe_sub == "sports_shoes":
            if top_sub != "tshirt" or bot_sub not in {"joggers", "track_pants", "shorts", "jeans"} or out_sub in (WESTERN_COATS_BLAZERS | {"jacket", "biker_jacket", "leather_jacket"}):
                return False, 0.0, 0.0, ["Incompatible: Athletic Gym Running Trainers only pair with T-Shirts & Joggers/Athleisure."]
            layer_shoe_score = min(1.0, (layer_shoe_score + 0.94) / 2.0)
            reasons.append("Footwear Harmony: High-performance Running Trainers matched with athletic ensemble.")

        elif shoe_sub in WESTERN_CASUAL_SHOES:
            if bot_sub in {"trousers", "formal_trousers"} or out_sub in WESTERN_COATS_BLAZERS or top_form == "formal":
                return False, 0.0, 0.0, ["Incompatible: Athletic/Casual Sneakers clash with Formal Suit Pants/Overcoat/Dress Shirt."]
            if top_sub in {"tshirt", "polo", "shirt"} and bot_sub in {"jeans", "cargos", "chinos", "shorts", "joggers"}:
                layer_shoe_score = min(1.0, (layer_shoe_score + 1.0) / 2.0 + 0.04)
                reasons.append("Footwear Harmony: Clean Sneakers ground the modern casual silhouette.")

        # Rule 3E: Loafers
        elif shoe_sub in WESTERN_SMART_SHOES:
            top_mat = str(top.get("material") or "").lower() if top else ""
            if bot_sub in {"cargos", "joggers", "track_pants"} or out_sub in {"puffer_jacket", "bomber_jacket"} or (
                top_sub == "tshirt" and (top_fit == "oversized" or top_pat == "graphic" or top_mat == "polyester")
            ):
                return False, 0.0, 0.0, ["Incompatible: Suede Penny Loafers clash with Gym Joggers/Puffer/Bomber/Graphic Tees."]
            layer_shoe_score = min(1.0, (layer_shoe_score + 0.96) / 2.0)
            reasons.append("Footwear Harmony: Suede Penny Loafers elevate the smart-casual profile.")

    # 4. MENSWEAR COLOR & SUIT COORDINATION RULES (No Green+Blue Coat, No Navy+Black Suit Clash, No Red+Olive Clash, No Brown Shoes+Black Suit)
    top_shade = str(top.get("shade") or top.get("primary_color") or "").lower() if top else ""
    bot_shade = str(bottom.get("shade") or bottom.get("primary_color") or "").lower() if bottom else ""
    out_shade = str(overtop.get("shade") or overtop.get("primary_color") or "").lower() if overtop else ""
    shoe_shade = str(shoe.get("shade") or shoe.get("primary_color") or "").lower() if shoe else ""

    top_col = str(top.get("primary_color") or "").lower() if top else ""
    bot_col = str(bottom.get("primary_color") or "").lower() if bottom else ""
    out_col = str(overtop.get("primary_color") or "").lower() if overtop else ""

    # Rule 4A: Never pair Green/Olive with Navy/Royal Blue Coat/Blazer/Shirt (or Red with Green/Olive)
    all_cols = {c for c in [top_col, bot_col, out_col] if c}
    all_shades = {s for s in [top_shade, bot_shade, out_shade] if s}
    if "red" in all_cols and ("green" in all_cols or "olive_green" in all_shades):
        return False, 0.0, 0.0, ["Incompatible Color Clash: Red and Olive/Green clash visually."]
    if ("green" in {top_col, out_col} or "emerald_green" in all_shades) and ("navy" in all_cols or "royal_blue" in all_shades):
        return False, 0.0, 0.0, ["Incompatible Color Clash: Green top/outerwear clashes with Navy/Royal Blue."]
    if out_col == "green" and top_col in {"blue", "navy", "red", "yellow", "maroon"}:
        return False, 0.0, 0.0, ["Incompatible Color Clash: Olive outerwear only pairs with neutral White, Black, Grey, or Beige tops."]

    # Rule 4B: Suit & Formal Trouser Color Coordination (No Navy Blazer + Black Suit Trousers, No Navy Shirt + Black Suit Trousers)
    if bot_sub in {"trousers", "formal_trousers"} and bot_shade == "black":
        if out_shade in {"navy", "dark_navy"} or top_shade in {"navy", "dark_navy"}:
            return False, 0.0, 0.0, ["Incompatible Menswear Clash: Navy Blazer/Shirt clashes with Jet Black Formal Suit Trousers."]
        if shoe_shade in {"chocolate_brown", "brown", "tan"}:
            return False, 0.0, 0.0, ["Incompatible Menswear Clash: Jet Black Formal Suit Trousers require Black Footwear (not Brown/Tan)."]

    if out_sub == "blazer" and out_shade in {"navy", "dark_navy"}:
        if bot_shade == "olive_green":
            return False, 0.0, 0.0, ["Incompatible Menswear Clash: Navy Suit Blazer pairs with Navy, Beige, Charcoal, or Indigo Denim (not Olive)."]
        if top_col not in {"white", "cream", "blue", "navy", "black", "grey"}:
            return False, 0.0, 0.0, ["Incompatible Menswear Clash: Navy Suit Blazer requires a Crisp White, Ivory, or Sky Blue shirt."]

    return True, float(top_bottom_score), float(layer_shoe_score), reasons


# ---------------------------------------------------------------------------
# Artificial Neural Network (PyTorch ANN) + ML Scorer (20 Features)
# ---------------------------------------------------------------------------
class OutfitCompatibilityANN(nn.Module):
    """
    Multi-Layer Perceptron (ANN) in PyTorch that maps the 20-dimensional
    4-Pillar + Sartorial Domain outfit compatibility feature vector to a calibrated
    probability score [0..1].
    Architecture:
      Input (20) -> Linear(20, 64) -> LayerNorm -> GELU -> Dropout(0.1)
                 -> Linear(64, 32) -> GELU
                 -> Linear(32, 16) -> GELU
                 -> Linear(16, 1)  -> Sigmoid
    """

    def __init__(self, in_features: int = 20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuralOutfitRecommender:
    _instance = None

    def __init__(self):
        self.device = torch.device("cpu")
        self.ann = OutfitCompatibilityANN(in_features=20).to(self.device)
        self.ml_gbm = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42
        )
        self._init_or_train_models()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _generate_training_data(self, n_samples: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates structured training samples across the 20-dimensional 4-Pillar + Sartorial feature space:
          f[0]: neutral_anchor_score (0..1)
          f[1]: complementary_harmony_score (0..1)
          f[2]: single_color_pop_score (0..1)
          f[3]: fit_harmony_score (0..1)
          f[4]: pattern_balance_score (0..1)
          f[5]: sartorial_top_bottom_score (0..1)  <-- Cultural/Garment Pairing (Kurta+Pajama vs Tshirt+Churidar)
          f[6]: sartorial_layer_shoe_score (0..1)  <-- Layer/Footwear Pairing (Kurta+Nehru+Mojari vs Kurta+Coat)
          f[7]: weather_active (0 or 1)
          f[8]: weather_material_layer_score (0..1)
          f[9]: weather_outerwear_trigger_score (0..1)
          f[10]: seasonal_alignment_score (-0.5..1.0)
          f[11]: occasion_active (0 or 1)
          f[12]: occasion_formality_match (0..1)
          f[13]: occasion_category_structure_match (0..1)
          f[14]: occasion_style_tag_match (0..1)
          f[15]: astrology_active (0 or 1)
          f[16]: astrology_birth_color_match (0..1)
          f[17]: astrology_weekday_color_match (0..1)
          f[18]: completeness_score (0..1)
          f[19]: layering_coherence_score (0..1)
        """
        rng = np.random.default_rng(42)
        X = np.zeros((n_samples, 20), dtype=np.float32)
        y = np.zeros(n_samples, dtype=np.float32)

        for i in range(n_samples):
            # Pillar 3 Color, Fit & Sartorial Compatibility (Always ON)
            X[i, 0] = rng.uniform(0.2, 1.0)
            X[i, 1] = rng.uniform(0.2, 1.0)
            X[i, 2] = rng.choice([0.3, 0.75, 1.0], p=[0.2, 0.3, 0.5])
            X[i, 3] = rng.uniform(0.4, 1.0)
            X[i, 4] = rng.choice([0.35, 0.95, 1.0], p=[0.2, 0.4, 0.4])

            # Sartorial Top+Bottom & Layer+Footwear compatibility (with 25% hard negative clashes)
            is_clash = rng.random() < 0.25
            if is_clash:
                X[i, 5] = rng.choice([0.0, 0.15])
                X[i, 6] = rng.choice([0.0, 0.15])
            else:
                X[i, 5] = rng.choice([0.84, 0.96, 1.0], p=[0.2, 0.3, 0.5])
                X[i, 6] = rng.choice([0.85, 0.95, 1.0], p=[0.2, 0.3, 0.5])

            # Pillar 1 (Weather)
            w_act = float(rng.choice([0.0, 1.0]))
            X[i, 7] = w_act
            X[i, 8] = rng.uniform(0.1, 1.0) if w_act else 0.0
            X[i, 9] = rng.uniform(0.2, 1.0) if w_act else 0.0
            X[i, 10] = rng.uniform(-0.3, 1.0) if w_act else 0.0

            # Pillar 2 (Occasion)
            o_act = float(rng.choice([0.0, 1.0]))
            X[i, 11] = o_act
            X[i, 12] = rng.uniform(0.1, 1.0) if o_act else 0.0
            X[i, 13] = rng.uniform(0.1, 1.0) if o_act else 0.0
            X[i, 14] = rng.uniform(0.2, 1.0) if o_act else 0.0

            # Pillar 4 (Astrology)
            a_act = float(rng.choice([0.0, 1.0]))
            X[i, 15] = a_act
            X[i, 16] = rng.uniform(0.0, 1.0) if a_act else 0.0
            X[i, 17] = rng.uniform(0.0, 1.0) if a_act else 0.0

            # Completeness & Layering
            X[i, 18] = rng.choice([0.65, 0.88, 1.0], p=[0.15, 0.35, 0.50])
            X[i, 19] = 0.1 if is_clash else rng.uniform(0.75, 1.0)

            sartorial_gate = 0.55 * X[i, 5] + 0.45 * X[i, 6]
            p3_score = (
                0.22 * max(X[i, 0], X[i, 1])
                + 0.16 * X[i, 2]
                + 0.14 * X[i, 3]
                + 0.12 * X[i, 4]
                + 0.36 * sartorial_gate
            )
            p1_score = 0.45 * X[i, 8] + 0.30 * X[i, 9] + 0.25 * max(0.0, X[i, 10])
            p2_score = 0.45 * X[i, 12] + 0.35 * X[i, 13] + 0.20 * X[i, 14]
            p4_score = 0.60 * X[i, 16] + 0.40 * X[i, 17]
            struct_score = 0.65 * X[i, 18] + 0.35 * X[i, 19]

            total_w = 1.15 + 0.35 + (0.65 * w_act) + (0.80 * o_act) + (0.55 * a_act)
            weighted_sum = (
                1.15 * p3_score
                + 0.35 * struct_score
                + (0.65 * w_act * p1_score)
                + (0.80 * o_act * p2_score)
                + (0.55 * a_act * p4_score)
            )
            raw_target = weighted_sum / total_w
            # Multiplicative penalty if sartorial pairing is broken (e.g., T-Shirt + Churidar or Kurta + Coat)
            if X[i, 5] < 0.3 or X[i, 6] < 0.3:
                raw_target *= 0.08
            y[i] = np.clip(raw_target, 0.02, 0.99)

        return X, y

    def _init_or_train_models(self):
        X, y = self._generate_training_data(n_samples=2000)

        # 1. Fit ML Gradient Boosting model
        self.ml_gbm.fit(X, y)

        # 2. Load or train PyTorch ANN
        if ANN_WEIGHTS_PATH.exists():
            try:
                self.ann.load_state_dict(torch.load(str(ANN_WEIGHTS_PATH), map_location=self.device))
                self.ann.eval()
                logger.info("[Recommender] Loaded pre-trained 20-feature PyTorch OutfitCompatibilityANN weights.")
                return
            except Exception:
                pass

        logger.info("[Recommender] Training 20-feature PyTorch OutfitCompatibilityANN (4-Pillar + Sartorial Neural Engine)...")
        self.ann.train()
        optimizer = torch.optim.AdamW(self.ann.parameters(), lr=0.008, weight_decay=1e-4)
        criterion = nn.MSELoss()

        x_tensor = torch.from_numpy(X).to(self.device)
        y_tensor = torch.from_numpy(y).unsqueeze(1).to(self.device)

        for _ in range(220):
            optimizer.zero_grad()
            preds = self.ann(x_tensor)
            loss = criterion(preds, y_tensor)
            loss.backward()
            optimizer.step()

        self.ann.eval()
        torch.save(self.ann.state_dict(), str(ANN_WEIGHTS_PATH))
        logger.info(f"[Recommender] Trained PyTorch ANN (final MSE loss: {loss.item():.5f}).")

    # -----------------------------------------------------------------------
    # Pillar Evaluation Methods
    # -----------------------------------------------------------------------
    def _evaluate_pillar3_color_and_fit(self, items: List[Dict[str, Any]]) -> Tuple[float, List[float], List[str]]:
        """
        Pillar 3: Color Theory, Fit & Sartorial Combination Matrix (Always ON - Main Pillar).
        Checks:
        1. Sartorial & Cultural Pairing Compatibility (Top+Bottom, Overtop, Footwear)
        2. Neutral Anchor Principle
        3. Complementary Harmony
        4. Single Color Pop Rule
        5. Fit Proportion & Pattern Balance
        """
        is_valid, sart_tb_score, sart_ls_score, sart_reasons = evaluate_sartorial_compatibility(items)
        reasons = list(sart_reasons)

        shades = [str(it.get("shade") or it.get("color", {}).get("shade") or "white").lower() for it in items]
        primaries = [str(it.get("primary_color") or it.get("color", {}).get("primary") or "white").lower() for it in items]
        all_color_tokens = set(shades + primaries)

        # 1. Neutral Anchor Principle
        neutral_count = sum(
            1 for s, p in zip(shades, primaries) if s in NEUTRAL_COLORS or p in NEUTRAL_COLORS
        )
        n_items = max(1, len(items))
        if neutral_count >= n_items - 1:
            neutral_anchor_score = 1.0
            reasons.append("Neutral Anchor Principle: Grounded by timeless neutrals for 100% color versatility.")
        elif neutral_count >= 1:
            neutral_anchor_score = 0.86
        else:
            neutral_anchor_score = 0.45

        # 2. Complementary Harmony on Color Wheel
        comp_score = 0.75
        for group_a, group_b in COMPLEMENTARY_PAIRS:
            if (all_color_tokens & group_a) and (all_color_tokens & group_b):
                comp_score = 1.0
                reasons.append("Complementary Color Harmony: High-contrast pairing across the fashion color wheel.")
                break
        if neutral_count == n_items:
            comp_score = 0.95

        # 3. Single Color Pop Rule
        vibrant_count = sum(
            1 for s, p in zip(shades, primaries)
            if (s in VIBRANT_POP_COLORS or p in VIBRANT_POP_COLORS) and (s not in NEUTRAL_COLORS and p not in NEUTRAL_COLORS)
        )
        if vibrant_count == 1 and neutral_count >= 1:
            pop_score = 1.0
            reasons.append("Single Color Pop Rule: 1 vibrant statement piece anchored by neutral coordinates.")
        elif vibrant_count == 0:
            pop_score = 0.92
        elif vibrant_count == 2 and neutral_count >= 1:
            pop_score = 0.68
        else:
            pop_score = 0.30
            reasons.append("Multiple vibrant colors compete visually.")

        # 4. Fit & Silhouette Harmony
        top_fit = next((it.get("fit_type", "regular") for it in items if it.get("category") == "top"), "regular")
        bot_fit = next((it.get("fit_type", "regular") for it in items if it.get("category") == "bottom"), "regular")
        has_structured_outer = any(it.get("subcategory") in {"blazer", "coat", "nehru_jacket"} for it in items)

        if has_structured_outer and bot_fit in ["tailored", "regular", "slim", "straight"]:
            fit_score = 1.0
            reasons.append(f"Fit Harmony: Structured outer layer balanced with {bot_fit} bottom silhouette.")
        elif (top_fit in ["oversized", "loose", "baggy", "relaxed"] and bot_fit in ["slim", "skinny", "regular", "tailored", "baggy"]) or (
            top_fit in ["slim", "skinny", "regular", "tailored"] and bot_fit in ["baggy", "oversized", "regular", "slim", "tailored", "straight"]
        ):
            fit_score = 0.96
            reasons.append(f"Fit Harmony: Well-proportioned {top_fit} top paired with {bot_fit} bottom.")
        else:
            fit_score = 0.80

        # 5. Pattern Balance
        patterns = [str(it.get("pattern") or "plain").lower() for it in items if it.get("category") in {"top", "bottom", "overtop"}]
        non_plain = [p for p in patterns if p not in ["plain", "solid"]]
        if len(non_plain) == 1:
            pattern_score = 1.0
            reasons.append(f"Pattern Balance: Single '{non_plain[0].replace('_', ' ')}' statement piece paired with clean solids.")
        elif len(non_plain) == 0:
            pattern_score = 0.95
        elif len(non_plain) == 2 and "ethnic_motif" in non_plain and "embroidered" in non_plain:
            pattern_score = 0.88
            reasons.append("Festive Layering: Coordinated embroidered kurta with brocade Nehru jacket.")
        else:
            pattern_score = 0.40
            reasons.append("Pattern Clash Notice: Multiple patterned garments in one outfit.")

        sartorial_avg = 0.55 * sart_tb_score + 0.45 * sart_ls_score
        pillar3_total = (
            0.22 * max(neutral_anchor_score, comp_score)
            + 0.16 * comp_score
            + 0.14 * pop_score
            + 0.12 * fit_score
            + 0.10 * pattern_score
            + 0.26 * sartorial_avg
        )
        if not is_valid:
            pillar3_total *= 0.10

        sub_feats = [
            neutral_anchor_score,
            comp_score,
            pop_score,
            fit_score,
            pattern_score,
            sart_tb_score,
            sart_ls_score,
        ]
        return float(np.clip(pillar3_total, 0.0, 1.0)), sub_feats, reasons

    def _evaluate_pillar1_weather(
        self, items: List[Dict[str, Any]], weather_info: Dict[str, Any]
    ) -> Tuple[float, List[float], List[str]]:
        """
        Pillar 1: Weather & Seasonal Compatibility Filter.
        - Temp > 28°C (Hot): Filter for material: cotton/linen, layering: none/base, category: top/shorts/light bottoms.
        - Temp < 15°C (Cold): Trigger outerwear pairing (category: overtop, material: wool/denim/leather/tweed/knit/silk).
        - Seasonal Alignment: Matching season adds +1.0 weight; conflicting items penalized.
        """
        reasons = []
        temp_c = float(weather_info.get("temperature_c", 28.0))
        regime = weather_info.get("regime", "mild")
        target_season = weather_info.get("season", "summer")

        cats = [it.get("category") for it in items]
        materials = [str(it.get("material") or "cotton").lower() for it in items]
        has_overtop = "overtop" in cats

        if temp_c >= 26.0 or regime == "hot":
            breathable_count = sum(
                1 for it, m in zip(items, materials)
                if m in ["cotton", "linen", "chiffon_georgette", "silk"] or it.get("subcategory") == "trousers"
            )
            heavy_count = sum(
                1 for it, m in zip(items, materials)
                if (m in ["velvet", "tweed", "corduroy"]) or (m in ["wool", "leather"] and it.get("category") == "overtop")
            )
            mat_score = np.clip((breathable_count / max(1, len(items))) - 0.35 * heavy_count, 0.1, 1.0)
            if has_overtop:
                outer_trigger_score = 0.0
                mat_score *= 0.25
                reasons.append(f"Weather ({temp_c:.1f}°C Hot): Outer jacket/coat strictly penalized in warm/hot weather.")
            else:
                outer_trigger_score = 1.0
                reasons.append(f"Weather ({temp_c:.1f}°C Hot): Breathable cotton/linen single base layer selected (no jacket).")
        elif temp_c < 15.0 or regime == "cold":
            warm_mats = sum(1 for m in materials if m in ["wool", "denim", "leather", "tweed", "knit", "polyester", "velvet", "silk"])
            mat_score = np.clip(0.5 + 0.25 * warm_mats, 0.2, 1.0)
            if has_overtop:
                outer_trigger_score = 1.0
                reasons.append(f"Weather ({temp_c:.1f}°C Cold): Warm outerwear layer paired for <15°C cold protection.")
            else:
                outer_trigger_score = 0.12
                reasons.append(f"Weather ({temp_c:.1f}°C Cold): Missing outerwear layer for <15°C cold weather.")
        elif temp_c >= 20.0:
            mat_score = 0.94
            if has_overtop:
                outer_trigger_score = 0.20
                reasons.append(f"Weather ({temp_c:.1f}°C Warm): Heavy layering avoided for comfortable warm weather.")
            else:
                outer_trigger_score = 1.0
                reasons.append(f"Weather ({temp_c:.1f}°C Warm): Single breathable layer matched for {weather_info.get('city', 'today')}.")
        else:
            # 15°C <= temp_c < 20°C (Cool / Pleasant)
            mat_score = 0.92
            outer_trigger_score = 0.95 if has_overtop else 0.85
            reasons.append(f"Weather ({temp_c:.1f}°C Cool): Light layering suitable for {weather_info.get('city', 'today')}.")

        season_weights = []
        for it in items:
            it_seasons = [s.lower() for s in (it.get("weather", {}).get("seasons") or [])]
            if target_season.lower() in it_seasons:
                season_weights.append(1.0)
            elif not it_seasons:
                season_weights.append(0.5)
            else:
                season_weights.append(-0.25)
        seasonal_score = float(np.clip(np.mean(season_weights), -0.5, 1.0))
        seasonal_norm = float(np.clip((seasonal_score + 0.5) / 1.5, 0.0, 1.0))

        pillar1_total = 0.45 * float(mat_score) + 0.30 * float(outer_trigger_score) + 0.25 * seasonal_norm
        return float(np.clip(pillar1_total, 0.0, 1.0)), [float(mat_score), float(outer_trigger_score), seasonal_score], reasons

    def _evaluate_pillar2_occasion(
        self, items: List[Dict[str, Any]], target_occasion: str
    ) -> Tuple[float, List[float], List[str]]:
        """
        Pillar 2: Event & Formality Context Filter.
        - Work / Formal Meeting (`work_formal` / `work_office`): Strict filter for formality: formal/smart_casual,
          category: shirt + trousers/chinos + optional blazer/coat.
        - Casual Outing / College (`casual_outing`, `college`): Allows style_tags: streetwear, casual,
          category: tshirt/polo/casual shirt + jeans/cargos/chinos.
        - Festival / Wedding (`festival`, `wedding`): Prioritizes Ethnic Kurta + Pajama/Churidar + Nehru Jacket + Mojari.
        """
        reasons = []
        occ = (target_occasion or "casual_outing").strip().lower()
        subcats = [str(it.get("subcategory") or "").lower() for it in items]
        formalities = [str(it.get("formality") or it.get("occasion", {}).get("formality") or "casual").lower() for it in items]
        all_style_tags = set()
        for it in items:
            for t in it.get("style_tags", []):
                all_style_tags.add(str(t).lower())

        has_ethnic_piece = any(s in (ETHNIC_TOPS | ETHNIC_BOTTOMS | ETHNIC_OUTERWEAR) for s in subcats)

        if occ in ["work_formal", "work_office", "formal", "work"]:
            if has_ethnic_piece or any(
                s in {"tshirt", "cargos", "shorts", "joggers", "biker_jacket", "jacket", "bomber_jacket", "puffer_jacket", "sneakers", "sports_shoes"}
                for s in subcats
            ):
                form_score = 0.15
                struct_score = 0.15
                style_score = 0.20
                reasons.append("Event Filter (Work / Formal): Casual/Ethnic/Athletic pieces penalized for strict corporate dress code.")
            else:
                valid_form = sum(1 for f in formalities if f in ["formal", "smart_casual", "semi_formal"])
                form_score = valid_form / max(1, len(items))
                has_shirt_and_trousers = ("shirt" in subcats) and any(s in {"trousers", "formal_trousers", "chinos"} for s in subcats)
                struct_score = 1.0 if has_shirt_and_trousers else 0.55
                style_score = 1.0 if ("formal" in all_style_tags or "classic" in all_style_tags) else 0.65
                reasons.append("Event Filter (Work / Formal): Prioritized tailored Shirt + Trousers/Chinos & formal dress code.")

        elif occ in ["casual_outing", "college", "casual"]:
            is_gym_piece = any(
                "gymwear" in [str(x).lower() for x in it.get("style_tags", [])]
                for it in items
            )
            if any(s in {"churidar", "pajama", "coat", "nehru_jacket", "oxford", "mojari", "sports_shoes", "joggers", "track_pants"} for s in subcats) or is_gym_piece:
                form_score = 0.22
                struct_score = 0.22
                style_score = 0.25
            else:
                valid_form = sum(1 for f in formalities if f in ["casual", "smart_casual"])
                form_score = valid_form / max(1, len(items))
                has_casual_pair = any(s in {"tshirt", "polo", "shirt"} for s in subcats) and any(
                    s in {"jeans", "cargos", "chinos", "shorts"} for s in subcats
                )
                struct_score = 1.0 if has_casual_pair else 0.55
                style_score = 1.0 if ("casual" in all_style_tags or "streetwear" in all_style_tags) else 0.70
                reasons.append(f"Event Filter ({occ.replace('_', ' ').title()}): Matched relaxed casual & streetwear separates.")

        elif occ in ["festival", "wedding", "ethnic"]:
            has_kurta_pajama = any(s in ETHNIC_TOPS for s in subcats) and any(s in ETHNIC_BOTTOMS for s in subcats)
            if has_kurta_pajama:
                form_score = 1.0
                struct_score = 1.0
                style_score = 1.0
                reasons.append(f"Event Filter ({occ.title()}): Traditional Kurta + Pajama/Churidar ensemble selected for celebration.")
            elif occ == "wedding" and ("shirt" in subcats and "trousers" in subcats and "blazer" in subcats):
                form_score = 0.90
                struct_score = 0.88
                style_score = 0.85
                reasons.append("Event Filter (Wedding): Tailored Blazer + Shirt + Trousers formal reception ensemble.")
            else:
                form_score = 0.25
                struct_score = 0.25
                style_score = 0.25

        elif occ in ["gym_sports", "gym", "sports", "athleisure"]:
            if has_ethnic_piece or any(
                s in {"shirt", "trousers", "formal_trousers", "chinos", "coat", "blazer", "jacket", "biker_jacket", "oxford", "loafers"}
                for s in subcats
            ):
                form_score = 0.08
                struct_score = 0.08
                style_score = 0.10
                reasons.append("Event Filter (Gym / Sports): Non-athletic formal/tailored pieces excluded for workout performance.")
            else:
                has_gym_pair = ("tshirt" in subcats) and any(s in {"joggers", "track_pants", "shorts"} for s in subcats)
                has_gym_shoes = any(s in {"sports_shoes", "sneakers"} for s in subcats)
                gym_tag_hits = sum(
                    1 for it in items
                    if "gym_sports" in [o.lower() for o in (it.get("occasion", {}).get("types") or [])]
                    or any(t in {"sporty", "athletic", "gymwear"} for t in [str(x).lower() for x in it.get("style_tags", [])])
                )
                form_score = max(0.75, gym_tag_hits / max(1, len(items)))
                struct_score = 1.0 if (has_gym_pair and has_gym_shoes) else (0.85 if has_gym_pair else 0.35)
                style_score = 1.0 if ("sporty" in all_style_tags or "athletic" in all_style_tags or "gymwear" in all_style_tags) else 0.65
                reasons.append("Event Filter (Gym / Sports): High-performance Dry-Fit Tee + Athletic Joggers + Running Trainers matched.")

        elif occ in ["party", "party_club", "date_night"]:
            is_gym_piece = any("gymwear" in [str(x).lower() for x in it.get("style_tags", [])] for it in items)
            if has_ethnic_piece or is_gym_piece or any(s in {"churidar", "pajama", "joggers", "track_pants", "sports_shoes", "puffer_jacket"} for s in subcats):
                form_score = 0.15
                struct_score = 0.15
                style_score = 0.15
            else:
                occ_matches = sum(
                    1 for it in items
                    if occ in [o.lower() for o in (it.get("occasion", {}).get("types") or [])]
                    or "party" in [o.lower() for o in (it.get("occasion", {}).get("types") or [])]
                    or "date_night" in [o.lower() for o in (it.get("occasion", {}).get("types") or [])]
                )
                form_score = max(0.60, occ_matches / max(1, len(items)))
                struct_score = 0.96
                style_score = 0.94
                reasons.append(f"Event Filter ({occ.replace('_', ' ').title()}): Elevated evening/date-night styling matched.")

        else:
            occ_matches = sum(
                1 for it in items if occ in [o.lower() for o in (it.get("occasion", {}).get("types") or [])]
            )
            form_score = max(0.55, occ_matches / max(1, len(items)))
            struct_score = 0.88
            style_score = 0.85
            reasons.append(f"Event Filter ({occ.replace('_', ' ').title()}): Scored pieces for occasion alignment.")

        pillar2_total = 0.45 * form_score + 0.35 * struct_score + 0.20 * style_score
        return float(np.clip(pillar2_total, 0.0, 1.0)), [float(form_score), float(struct_score), float(style_score)], reasons

    def _evaluate_pillar4_astrology(
        self, items: List[Dict[str, Any]], astro_info: Dict[str, Any]
    ) -> Tuple[float, List[float], List[str]]:
        """
        Pillar 4: Personalized Date of Birth (DOB) Lucky Color Engine.
        """
        reasons = []
        birth_lucky = set(astro_info.get("birth_lucky_shades", []))
        weekday_lucky = set(astro_info.get("weekday_lucky_shades", []))

        item_colors = set()
        matched_names = []
        for it in items:
            sh = str(it.get("shade") or it.get("color", {}).get("shade") or "").lower()
            pr = str(it.get("primary_color") or it.get("color", {}).get("primary") or "").lower()
            item_colors.add(sh)
            item_colors.add(pr)
            if sh in birth_lucky or pr in birth_lucky or sh in weekday_lucky or pr in weekday_lucky:
                matched_names.append(f"{it['name']} ({sh.replace('_', ' ')})")

        birth_hit = len(item_colors & birth_lucky) > 0
        weekday_hit = len(item_colors & weekday_lucky) > 0

        birth_score = 1.0 if birth_hit else 0.25
        weekday_score = 1.0 if weekday_hit else 0.25

        if birth_hit and weekday_hit:
            pillar4_total = 1.0
            reasons.append(
                f"Astrology Lucky Color Match: {', '.join(matched_names[:2])} aligns with both "
                f"Birth #{astro_info['birth_number']} ({astro_info['birth_planet']}) and {astro_info['weekday']} ({astro_info['weekday_planet']})."
            )
        elif birth_hit or weekday_hit:
            pillar4_total = 0.85
            reasons.append(
                f"Astrology Lucky Color Match: {', '.join(matched_names[:2])} aligns with your planetary lucky palette ({astro_info['summary']})."
            )
        else:
            pillar4_total = 0.30
            reasons.append(
                f"Astrology Note: No exact lucky shade from {astro_info['summary']} found in this combination."
            )

        return float(pillar4_total), [float(birth_score), float(weekday_score)], reasons

    # -----------------------------------------------------------------------
    # Main Recommendation Entrypoint
    # -----------------------------------------------------------------------
    def recommend_outfits(
        self,
        wardrobe_items: List[Dict[str, Any]],
        use_weather: bool = False,
        use_occasion: bool = False,
        use_astrology: bool = False,
        use_color_fit_harmony: bool = True,  # Always ON
        occasion: str = "casual_outing",
        dob: Optional[str] = "2003-05-15",
        city: Optional[str] = "Mumbai",
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        override_temp_c: Optional[float] = None,
        top_k: int = 4,
    ) -> Dict[str, Any]:
        """
        Generates sartorially valid candidate outfits from the user's wardrobe and ranks them
        using the 4-Pillar Algorithmic Model + 20-feature PyTorch ANN + ML Gradient Boosting ensemble.
        """
        use_color_fit_harmony = True

        live_weather = fetch_today_weather(city=city, latitude=latitude, longitude=longitude, override_temp_c=override_temp_c)
        ambient_temp = float(live_weather.get("temperature_c", 29.0))
        weather_info = live_weather if use_weather else None
        astro_info = calculate_astrology_lucky_colors(dob_str=dob) if use_astrology else None

        if not wardrobe_items:
            return {
                "status": "EMPTY_WARDROBE",
                "message": "Your wardrobe is empty. Upload an outfit photo and click 'Add to Wardrobe' first!",
                "active_pillars": {
                    "color_fit_harmony": True,
                    "weather": use_weather,
                    "occasion": use_occasion,
                    "astrology": use_astrology,
                },
                "weather_context": weather_info,
                "astrology_context": astro_info,
                "occasion_context": occasion if use_occasion else None,
                "recommendations": [],
            }

        tops = [it for it in wardrobe_items if it.get("category") == "top"]
        bottoms = [it for it in wardrobe_items if it.get("category") == "bottom"]
        dresses = [it for it in wardrobe_items if it.get("category") == "dress"]
        overtops = [it for it in wardrobe_items if it.get("category") == "overtop"]
        footwear = [it for it in wardrobe_items if it.get("category") == "footwear"]
        accessories = [it for it in wardrobe_items if it.get("category") == "accessory"]

        candidates: List[List[Dict[str, Any]]] = []

        # Build sartorially valid combinations of Top + Bottom (+ optional Overtop, Footwear, Accessory)
        if tops and bottoms:
            # Never pair jackets/coats/puffers/bombers when temperature >= 22°C (e.g. 29°C warm/hot weather)!
            # In cold weather (< 15°C) when Weather Pillar is active, require outerwear if available.
            if ambient_temp >= 22.0:
                if not use_weather and use_occasion and occasion in ["work_formal", "wedding"]:
                    # Allow only tailored suit blazer/nehru_jacket if Weather pillar is OFF and formal/wedding occasion is ON
                    allowed_outers = [o for o in overtops if o.get("subcategory") in {"blazer", "nehru_jacket"}]
                    outer_opts = [[]] + [[o] for o in allowed_outers]
                else:
                    outer_opts = [[]]
            elif use_weather and ambient_temp < 15.0 and overtops:
                outer_opts = [[o] for o in overtops] + [[]]
            else:
                outer_opts = [[]] + [[o] for o in overtops] if overtops else [[]]
            shoe_opts = [[]] + [[f] for f in footwear] if footwear else [[]]
            acc_opts = [[a] for a in accessories[:2]] if accessories else [[]]

            for t, b in itertools.product(tops, bottoms):
                # Early prune incompatible Top + Bottom (e.g., T-Shirt + Churidar, Shirt + Pajama, Kurta + Cargos)
                valid_tb, _, _, _ = evaluate_sartorial_compatibility([t, b])
                if not valid_tb:
                    continue

                for ot in outer_opts:
                    # Early prune incompatible Outerwear (e.g., Western Coat / Biker Jacket over Kurta)
                    if ot:
                        valid_tbo, _, _, _ = evaluate_sartorial_compatibility([t, b] + ot)
                        if not valid_tbo:
                            continue

                    for sh in shoe_opts:
                        # If footwear exists in wardrobe, prefer complete outfits with shoes
                        if footwear and not sh:
                            continue
                        valid_full, _, _, _ = evaluate_sartorial_compatibility([t, b] + ot + sh)
                        if not valid_full:
                            continue

                        for ac in acc_opts[:1]:
                            candidates.append([t, b] + ot + sh + ac)

        # Build combinations for One-Piece Dresses / Ethnic Sets
        for d in dresses:
            shoe_opts = [[f] for f in footwear[:3]] if footwear else [[]]
            outer_opts = [[]] + [[o] for o in overtops[:2]] if overtops else [[]]
            for sh in shoe_opts:
                for ot in outer_opts:
                    valid_d, _, _, _ = evaluate_sartorial_compatibility([d] + ot + sh)
                    if valid_d:
                        candidates.append([d] + ot + sh)

        # Fallback if user only has single category items
        if not candidates:
            for item in wardrobe_items[:6]:
                candidates.append([item])

        scored_results = []
        for combo in candidates:
            cats = [it.get("category") for it in combo]
            has_top = "top" in cats
            has_bottom = "bottom" in cats
            has_dress = "dress" in cats
            has_shoes = "footwear" in cats
            has_outer = "overtop" in cats

            if (has_top and has_bottom and has_shoes) or (has_dress and has_shoes):
                completeness = 1.0
            elif (has_top and has_bottom) or has_dress:
                completeness = 0.88
            else:
                completeness = 0.60

            layering_coherence = 1.0 if (not has_outer or has_top or has_dress) else 0.70

            # Pillar 3: Color, Fit & Sartorial Harmony (Always ON)
            p3_score, p3_feats, p3_reasons = self._evaluate_pillar3_color_and_fit(combo)

            # Pillar 1: Weather & Season (if ticked)
            if use_weather and weather_info:
                p1_score, p1_feats, p1_reasons = self._evaluate_pillar1_weather(combo, weather_info)
                w_flag = 1.0
            else:
                p1_score, p1_feats, p1_reasons = 0.0, [0.0, 0.0, 0.0], []
                w_flag = 0.0

            # Pillar 2: Event & Occasion (if ticked)
            if use_occasion:
                p2_score, p2_feats, p2_reasons = self._evaluate_pillar2_occasion(combo, occasion)
                o_flag = 1.0
            else:
                p2_score, p2_feats, p2_reasons = 0.0, [0.0, 0.0, 0.0], []
                o_flag = 0.0

            # Pillar 4: Astrology / DOB Lucky Color (if ticked)
            if use_astrology and astro_info:
                p4_score, p4_feats, p4_reasons = self._evaluate_pillar4_astrology(combo, astro_info)
                a_flag = 1.0
            else:
                p4_score, p4_feats, p4_reasons = 0.0, [0.0, 0.0], []
                a_flag = 0.0

            # Assemble 20-d feature vector for PyTorch ANN + ML GBM
            feat_vec = np.array(
                p3_feats
                + [w_flag] + p1_feats
                + [o_flag] + p2_feats
                + [a_flag] + p4_feats
                + [completeness, layering_coherence],
                dtype=np.float32,
            )

            with torch.no_grad():
                t_in = torch.from_numpy(feat_vec).unsqueeze(0).to(self.device)
                ann_prob = float(self.ann(t_in).item())

            ml_prob = float(np.clip(self.ml_gbm.predict(feat_vec.reshape(1, -1))[0], 0.0, 1.0))

            active_weights = 1.15 + 0.25 + (0.65 * w_flag) + (0.85 * o_flag) + (0.55 * a_flag)
            algo_prob = (
                1.15 * p3_score
                + 0.25 * completeness
                + (0.65 * w_flag * p1_score)
                + (0.85 * o_flag * p2_score)
                + (0.55 * a_flag * p4_score)
            ) / active_weights

            final_prob = 0.50 * algo_prob + 0.30 * ann_prob + 0.20 * ml_prob
            if not use_occasion:
                has_gym_only = any(
                    it.get("subcategory") in {"sports_shoes", "joggers", "track_pants"}
                    or "gymwear" in [str(x).lower() for x in it.get("style_tags", [])]
                    for it in combo
                )
                if has_gym_only:
                    final_prob *= 0.88
            final_score = round(float(np.clip(final_prob * 100.0, 10.0, 99.5)), 1)

            all_reasons = p3_reasons + p1_reasons + p2_reasons + p4_reasons

            scored_results.append({
                "items": combo,
                "item_ids": [it["id"] for it in combo],
                "score": final_score,
                "ann_confidence": round(ann_prob * 100.0, 1),
                "ml_confidence": round(ml_prob * 100.0, 1),
                "pillar_scores": {
                    "pillar3_color_fit_harmony": round(p3_score * 100.0, 1),
                    "pillar1_weather_season": round(p1_score * 100.0, 1) if use_weather else None,
                    "pillar2_event_occasion": round(p2_score * 100.0, 1) if use_occasion else None,
                    "pillar4_astrology_lucky_color": round(p4_score * 100.0, 1) if use_astrology else None,
                },
                "reasons": all_reasons,
                "explanation": " • ".join(all_reasons[:4]),
            })

        scored_results.sort(key=lambda x: x["score"], reverse=True)

        # Multi-Slot Outfit Diversity: ensure the top_k recommendations use different tops,
        # bottoms, outerwear, and shoes rather than repeating the exact same pants/jacket/shoes!
        unique_outfits = []
        used_tops = set()
        used_bots = set()
        used_shoes = set()
        used_outers = set()
        seen_tb_pairs = set()

        def _extract_slot_ids(res_obj):
            t_id = next((it["id"] for it in res_obj["items"] if it.get("category") == "top"), 0)
            b_id = next((it["id"] for it in res_obj["items"] if it.get("category") == "bottom"), 0)
            s_id = next((it["id"] for it in res_obj["items"] if it.get("category") == "footwear"), 0)
            o_id = next((it["id"] for it in res_obj["items"] if it.get("category") == "overtop"), 0)
            return t_id, b_id, s_id, o_id

        # Pass 1: Strict Full-Slot Diversity among high-scoring candidates (every outfit has a different top, bottom, shoe, and jacket)
        top_score = scored_results[0]["score"] if scored_results else 0.0
        for res in scored_results:
            if res["score"] < top_score - 6.8:
                break
            t_id, b_id, s_id, o_id = _extract_slot_ids(res)
            if t_id and t_id in used_tops:
                continue
            if b_id and b_id in used_bots:
                continue
            if s_id and s_id in used_shoes:
                continue
            if o_id and o_id in used_outers:
                continue
            unique_outfits.append(res)
            if t_id:
                used_tops.add(t_id)
            if b_id:
                used_bots.add(b_id)
            if s_id:
                used_shoes.add(s_id)
            if o_id:
                used_outers.add(o_id)
            seen_tb_pairs.add((t_id, b_id))
            if len(unique_outfits) >= top_k:
                break

        # Pass 2: If a category has fewer items than top_k (e.g. 2 ethnic bottoms or 3 gym joggers),
        # still require a distinct top_id and distinct (top_id, bot_id), and cap bottom/shoe reuse to <= 2
        if len(unique_outfits) < top_k:
            bot_counts = {}
            shoe_counts = {}
            outer_counts = {}
            for u in unique_outfits:
                _, b_id, s_id, o_id = _extract_slot_ids(u)
                bot_counts[b_id] = bot_counts.get(b_id, 0) + 1
                shoe_counts[s_id] = shoe_counts.get(s_id, 0) + 1
                if o_id:
                    outer_counts[o_id] = outer_counts.get(o_id, 0) + 1

            for res in scored_results:
                if res["score"] < top_score - 8.0:
                    break
                t_id, b_id, s_id, o_id = _extract_slot_ids(res)
                if (t_id, b_id) in seen_tb_pairs:
                    continue
                if t_id and t_id in used_tops:
                    continue
                if b_id and bot_counts.get(b_id, 0) >= 2:
                    continue
                if s_id and shoe_counts.get(s_id, 0) >= 2:
                    continue
                if o_id and outer_counts.get(o_id, 0) >= 2:
                    continue
                unique_outfits.append(res)
                if t_id:
                    used_tops.add(t_id)
                bot_counts[b_id] = bot_counts.get(b_id, 0) + 1
                shoe_counts[s_id] = shoe_counts.get(s_id, 0) + 1
                if o_id:
                    outer_counts[o_id] = outer_counts.get(o_id, 0) + 1
                seen_tb_pairs.add((t_id, b_id))
                if len(unique_outfits) >= top_k:
                    break

        # Pass 3: Fallback to distinct (top_id, bot_id) pairs if wardrobe is small
        if len(unique_outfits) < top_k:
            for res in scored_results:
                t_id, b_id, _, _ = _extract_slot_ids(res)
                tb_key = (t_id, b_id)
                if tb_key not in seen_tb_pairs:
                    seen_tb_pairs.add(tb_key)
                    unique_outfits.append(res)
                if len(unique_outfits) >= top_k:
                    break

        # Pass 4: Fallback if fewer than top_k distinct (top, bottom) pairs exist in a tiny wardrobe
        if len(unique_outfits) < top_k:
            seen_full = {tuple(sorted(u["item_ids"])) for u in unique_outfits}
            for res in scored_results:
                fk = tuple(sorted(res["item_ids"]))
                if fk not in seen_full:
                    seen_full.add(fk)
                    unique_outfits.append(res)
                if len(unique_outfits) >= top_k:
                    break

        unique_outfits.sort(key=lambda x: x["score"], reverse=True)

        return {
            "status": "OK",
            "count": len(unique_outfits),
            "active_pillars": {
                "color_fit_harmony": True,
                "weather": use_weather,
                "occasion": use_occasion,
                "astrology": use_astrology,
            },
            "weather_context": weather_info,
            "astrology_context": astro_info,
            "occasion_context": occasion if use_occasion else None,
            "recommendations": unique_outfits,
        }
