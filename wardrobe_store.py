"""
Persistent SQLite Store for User Profile (with DOB) and Digital Wardrobe Items.
Includes:
1. Intelligent duplicate detection so adding an item already present in the
   wardrobe is automatically recognized as 'Already in Wardrobe'.
2. Predefined Starter Wardrobe of 30 photorealistic Nano Banana 3D ghost-mannequin
   studio garments & footwear across T-Shirts, Shirts, Kurtas, Pajamas, Churidars,
   Trousers, Jeans, Chinos, Joggers, Winter Overcoats, Blazers, Leather Biker Jackets,
   Winter Puffer Jackets, Bomber Jackets, Sneakers, Loafers, Oxfords, Sandals, and Bags.
"""

import os
import json
import sqlite3
import hashlib
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

DB_PATH = Path(__file__).resolve().parent / "wardrobe.db"
STATIC_DIR = Path(__file__).resolve().parent / "static"
PREDEFINED_DIR = STATIC_DIR / "predefined_wardrobe"

SHADE_HEX_MAP = {
    "pure_white": "#FFFFFF",
    "white": "#FAFAFA",
    "off_white": "#F4F1EA",
    "cream": "#FFFAA0",
    "beige": "#E8DCC4",
    "tan": "#D2B48C",
    "khaki": "#C3B091",
    "brown": "#6E4726",
    "black": "#18181B",
    "charcoal_grey": "#374151",
    "grey": "#6B7280",
    "light_grey": "#D1D5DB",
    "navy": "#172554",
    "dark_navy": "#0F172A",
    "royal_blue": "#1D4ED8",
    "sky_blue": "#7DD3FC",
    "indigo_denim": "#1E3A5F",
    "faded_denim": "#60A5FA",
    "blue": "#2563EB",
    "olive_green": "#4D5D28",
    "emerald_green": "#047857",
    "forest_green": "#14532D",
    "mint": "#A7F3D0",
    "teal": "#0F766E",
    "mustard": "#EAB308",
    "yellow": "#FACC15",
    "orange": "#EA580C",
    "red": "#DC2626",
    "maroon": "#7F1D1D",
    "burgundy": "#6B1128",
    "rust": "#9A3412",
    "pink": "#F472B6",
    "baby_pink": "#FBCFE8",
    "lavender": "#DDD6FE",
}


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Curated Predefined Starter Wardrobe (38 Real Nano Banana Men's Studio Packshots)
# ---------------------------------------------------------------------------
PREDEFINED_WARDROBE_CATALOG: List[Dict[str, Any]] = [
    # --- MEN'S CASUAL & STREETWEAR T-SHIRTS (4) ---
    {
        "slug": "classic_white_tshirt",
        "image_url": "/static/predefined_wardrobe/classic_white_tshirt.jpg",
        "name": "Classic Crisp White Crewneck T-Shirt",
        "category": "top",
        "subcategory": "tshirt",
        "primary_color": "white",
        "shade": "pure_white",
        "color_family": "neutral",
        "pattern": "plain",
        "material": "cotton",
        "fit_type": "regular",
        "silhouette": "straight",
        "length": "regular",
        "layer": "base",
        "formality": "casual",
        "occasions": ["casual_outing", "college", "vacation"],
        "seasons": ["summer", "spring", "monsoon"],
        "style_tags": ["casual", "minimal", "classic"],
    },
    {
        "slug": "black_oversized_street_tee",
        "image_url": "/static/predefined_wardrobe/black_oversized_street_tee.jpg",
        "name": "Jet Black 1973 Streetwear Graphic Tee",
        "category": "top",
        "subcategory": "tshirt",
        "primary_color": "black",
        "shade": "black",
        "color_family": "neutral",
        "pattern": "graphic",
        "material": "cotton",
        "fit_type": "oversized",
        "silhouette": "boxy",
        "length": "regular",
        "layer": "base",
        "formality": "casual",
        "occasions": ["casual_outing", "college", "party"],
        "seasons": ["summer", "spring", "autumn", "winter"],
        "style_tags": ["streetwear", "casual", "edgy"],
    },
    {
        "slug": "olive_street_tshirt",
        "image_url": "/static/predefined_wardrobe/olive_street_tshirt.jpg",
        "name": "Olive Green 1973 Vintage Cotton Tee",
        "category": "top",
        "subcategory": "tshirt",
        "primary_color": "green",
        "shade": "olive_green",
        "color_family": "earthy",
        "pattern": "graphic",
        "material": "cotton",
        "fit_type": "regular",
        "silhouette": "straight",
        "length": "regular",
        "layer": "base",
        "formality": "casual",
        "occasions": ["casual_outing", "college", "vacation"],
        "seasons": ["summer", "spring", "autumn"],
        "style_tags": ["streetwear", "casual"],
    },
    {
        "slug": "crimson_red_tshirt",
        "image_url": "/static/predefined_wardrobe/crimson_red_tshirt.png",
        "name": "Crimson Red Moto Graphic Cotton Tee",
        "category": "top",
        "subcategory": "tshirt",
        "primary_color": "red",
        "shade": "red",
        "color_family": "warm",
        "pattern": "graphic",
        "material": "cotton",
        "fit_type": "regular",
        "silhouette": "straight",
        "length": "regular",
        "layer": "base",
        "formality": "casual",
        "occasions": ["casual_outing", "college", "party"],
        "seasons": ["summer", "spring"],
        "style_tags": ["casual", "streetwear"],
    },
    # --- MEN'S GYMWEAR & ATHLEISURE TOPS (2) ---
    {
        "slug": "cobalt_gym_dryfit_tshirt",
        "image_url": "/static/predefined_wardrobe/cobalt_gym_dryfit_tshirt.jpg",
        "name": "Cobalt Blue Dry-Fit Gym Training Tee",
        "category": "top",
        "subcategory": "tshirt",
        "primary_color": "blue",
        "shade": "royal_blue",
        "color_family": "cool",
        "pattern": "graphic",
        "material": "polyester",
        "fit_type": "slim",
        "silhouette": "structured",
        "length": "regular",
        "layer": "base",
        "formality": "casual",
        "occasions": ["gym_sports"],
        "seasons": ["summer", "spring", "autumn", "winter"],
        "style_tags": ["sporty", "athletic", "gymwear"],
    },
    {
        "slug": "charcoal_gym_performance_tee",
        "image_url": "/static/predefined_wardrobe/charcoal_gym_performance_tee.jpg",
        "name": "Charcoal Grey Compression Gym Performance Tee",
        "category": "top",
        "subcategory": "tshirt",
        "primary_color": "grey",
        "shade": "charcoal_grey",
        "color_family": "neutral",
        "pattern": "graphic",
        "material": "polyester",
        "fit_type": "slim",
        "silhouette": "structured",
        "length": "regular",
        "layer": "base",
        "formality": "casual",
        "occasions": ["gym_sports"],
        "seasons": ["summer", "spring", "autumn", "winter"],
        "style_tags": ["sporty", "athletic", "gymwear"],
    },
    # --- MEN'S SHIRTS (4) ---
    {
        "slug": "sky_blue_oxford_shirt",
        "image_url": "/static/predefined_wardrobe/sky_blue_oxford_shirt.jpg",
        "name": "Sky Blue Oxford Tailored Formal Shirt",
        "category": "top",
        "subcategory": "shirt",
        "primary_color": "blue",
        "shade": "sky_blue",
        "color_family": "cool",
        "pattern": "plain",
        "material": "cotton",
        "fit_type": "tailored",
        "silhouette": "structured",
        "length": "regular",
        "layer": "base",
        "formality": "formal",
        "occasions": ["work_formal", "date_night", "wedding"],
        "seasons": ["summer", "spring", "autumn", "winter"],
        "style_tags": ["formal", "classic", "minimal"],
    },
    {
        "slug": "ivory_linen_shirt",
        "image_url": "/static/predefined_wardrobe/ivory_linen_shirt.jpg",
        "name": "Ivory White Relaxed Linen Resort Shirt",
        "category": "top",
        "subcategory": "shirt",
        "primary_color": "white",
        "shade": "off_white",
        "color_family": "neutral",
        "pattern": "plain",
        "material": "linen",
        "fit_type": "relaxed",
        "silhouette": "relaxed",
        "length": "regular",
        "layer": "base",
        "formality": "smart_casual",
        "occasions": ["work_formal", "casual_outing", "date_night", "vacation"],
        "seasons": ["summer", "spring"],
        "style_tags": ["minimal", "chic", "classic"],
    },
    {
        "slug": "navy_crisp_formal_shirt",
        "image_url": "/static/predefined_wardrobe/navy_crisp_formal_shirt.jpg",
        "name": "Midnight Navy Crisp Dress Shirt",
        "category": "top",
        "subcategory": "shirt",
        "primary_color": "navy",
        "shade": "navy",
        "color_family": "neutral",
        "pattern": "plain",
        "material": "cotton",
        "fit_type": "slim",
        "silhouette": "structured",
        "length": "regular",
        "layer": "base",
        "formality": "formal",
        "occasions": ["work_formal", "party", "date_night", "wedding"],
        "seasons": ["summer", "spring", "autumn", "winter"],
        "style_tags": ["formal", "chic"],
    },
    {
        "slug": "white_crisp_formal_shirt",
        "image_url": "/static/predefined_wardrobe/white_crisp_formal_shirt.jpg",
        "name": "Crisp Pure White Tailored Dress Shirt",
        "category": "top",
        "subcategory": "shirt",
        "primary_color": "white",
        "shade": "pure_white",
        "color_family": "neutral",
        "pattern": "plain",
        "material": "cotton",
        "fit_type": "tailored",
        "silhouette": "structured",
        "length": "regular",
        "layer": "base",
        "formality": "formal",
        "occasions": ["work_formal", "party", "date_night", "wedding"],
        "seasons": ["summer", "spring", "autumn", "winter"],
        "style_tags": ["formal", "classic", "minimal", "chic"],
    },
    # --- MEN'S ETHNIC KURTAS (4) ---
    {
        "slug": "royal_blue_silk_kurta",
        "image_url": "/static/predefined_wardrobe/royal_blue_silk_kurta.jpg",
        "name": "Royal Blue Silk Embroidered Festive Kurta",
        "category": "top",
        "subcategory": "kurta",
        "primary_color": "blue",
        "shade": "royal_blue",
        "color_family": "cool",
        "pattern": "embroidered",
        "material": "silk",
        "fit_type": "regular",
        "silhouette": "straight",
        "length": "long",
        "layer": "base",
        "formality": "semi_formal",
        "occasions": ["festival", "wedding"],
        "seasons": ["summer", "autumn", "winter", "spring"],
        "style_tags": ["ethnic", "traditional", "festive"],
    },
    {
        "slug": "mustard_cotton_kurta",
        "image_url": "/static/predefined_wardrobe/mustard_cotton_kurta.jpg",
        "name": "Mustard Yellow Handloom Cotton Kurta",
        "category": "top",
        "subcategory": "kurta",
        "primary_color": "yellow",
        "shade": "mustard",
        "color_family": "warm",
        "pattern": "plain",
        "material": "cotton",
        "fit_type": "regular",
        "silhouette": "straight",
        "length": "long",
        "layer": "base",
        "formality": "smart_casual",
        "occasions": ["festival", "casual_outing", "wedding"],
        "seasons": ["summer", "spring", "autumn"],
        "style_tags": ["ethnic", "indo_western", "traditional"],
    },
    {
        "slug": "maroon_wedding_kurta",
        "image_url": "/static/predefined_wardrobe/maroon_wedding_kurta.jpg",
        "name": "Deep Maroon Jacquard Silk Wedding Kurta",
        "category": "top",
        "subcategory": "kurta",
        "primary_color": "maroon",
        "shade": "maroon",
        "color_family": "warm",
        "pattern": "ethnic_motif",
        "material": "silk",
        "fit_type": "tailored",
        "silhouette": "structured",
        "length": "long",
        "layer": "base",
        "formality": "formal",
        "occasions": ["wedding", "festival"],
        "seasons": ["winter", "autumn", "spring"],
        "style_tags": ["ethnic", "traditional", "wedding"],
    },
    {
        "slug": "cream_chikankari_kurta",
        "image_url": "/static/predefined_wardrobe/cream_chikankari_kurta.jpg",
        "name": "Ivory Cream Handloom Festive Kurta",
        "category": "top",
        "subcategory": "kurta",
        "primary_color": "cream",
        "shade": "cream",
        "color_family": "neutral",
        "pattern": "plain",
        "material": "cotton",
        "fit_type": "relaxed",
        "silhouette": "straight",
        "length": "long",
        "layer": "base",
        "formality": "semi_formal",
        "occasions": ["festival", "wedding", "casual_outing"],
        "seasons": ["summer", "spring"],
        "style_tags": ["ethnic", "traditional", "minimal"],
    },
    # --- MEN'S ETHNIC PAJAMA & CHURIDAR (2) ---
    {
        "slug": "white_cotton_pajama",
        "image_url": "/static/predefined_wardrobe/white_cotton_pajama.jpg",
        "name": "Classic Pure White Cotton Kurta Pajama",
        "category": "bottom",
        "subcategory": "pajama",
        "primary_color": "white",
        "shade": "pure_white",
        "color_family": "neutral",
        "pattern": "plain",
        "material": "cotton",
        "fit_type": "regular",
        "silhouette": "straight",
        "length": "full_length",
        "layer": "base",
        "formality": "semi_formal",
        "occasions": ["festival", "wedding"],
        "seasons": ["summer", "spring", "autumn", "winter"],
        "style_tags": ["ethnic", "traditional"],
    },
    {
        "slug": "cream_silk_churidar",
        "image_url": "/static/predefined_wardrobe/cream_silk_churidar.jpg",
        "name": "Warm Cream Silk Tapered Churidar",
        "category": "bottom",
        "subcategory": "churidar",
        "primary_color": "cream",
        "shade": "cream",
        "color_family": "neutral",
        "pattern": "plain",
        "material": "silk",
        "fit_type": "slim",
        "silhouette": "tapered",
        "length": "full_length",
        "layer": "base",
        "formality": "formal",
        "occasions": ["wedding", "festival"],
        "seasons": ["summer", "spring", "autumn", "winter"],
        "style_tags": ["ethnic", "traditional", "wedding"],
    },
    # --- MEN'S FORMAL SUIT PANTS, CHINOS & JEANS (7) ---
    {
        "slug": "charcoal_tailored_trousers",
        "image_url": "/static/predefined_wardrobe/charcoal_tailored_trousers.jpg",
        "name": "Charcoal Grey Wool Formal Suit Pants",
        "category": "bottom",
        "subcategory": "trousers",
        "primary_color": "grey",
        "shade": "charcoal_grey",
        "color_family": "neutral",
        "pattern": "plain",
        "material": "wool",
        "fit_type": "tailored",
        "silhouette": "straight",
        "length": "full_length",
        "layer": "base",
        "formality": "formal",
        "occasions": ["work_formal", "wedding", "date_night"],
        "seasons": ["summer", "spring", "autumn", "winter"],
        "style_tags": ["formal", "classic", "minimal"],
    },
    {
        "slug": "navy_formal_suit_pants",
        "image_url": "/static/predefined_wardrobe/navy_formal_suit_pants.jpg",
        "name": "Midnight Navy Tailored Formal Suit Pants",
        "category": "bottom",
        "subcategory": "trousers",
        "primary_color": "navy",
        "shade": "navy",
        "color_family": "neutral",
        "pattern": "plain",
        "material": "wool",
        "fit_type": "tailored",
        "silhouette": "straight",
        "length": "full_length",
        "layer": "base",
        "formality": "formal",
        "occasions": ["work_formal", "wedding", "date_night", "party"],
        "seasons": ["summer", "spring", "autumn", "winter"],
        "style_tags": ["formal", "classic", "chic"],
    },
    {
        "slug": "black_formal_suit_pants",
        "image_url": "/static/predefined_wardrobe/black_formal_suit_pants.jpg",
        "name": "Jet Black Tailored Formal Suit Trousers",
        "category": "bottom",
        "subcategory": "trousers",
        "primary_color": "black",
        "shade": "black",
        "color_family": "neutral",
        "pattern": "plain",
        "material": "wool",
        "fit_type": "tailored",
        "silhouette": "straight",
        "length": "full_length",
        "layer": "base",
        "formality": "formal",
        "occasions": ["work_formal", "wedding", "date_night", "party"],
        "seasons": ["summer", "spring", "autumn", "winter"],
        "style_tags": ["formal", "classic", "minimal"],
    },
    {
        "slug": "beige_slim_chinos",
        "image_url": "/static/predefined_wardrobe/beige_slim_chinos.jpg",
        "name": "Warm Beige Slim Stretch Chinos",
        "category": "bottom",
        "subcategory": "chinos",
        "primary_color": "beige",
        "shade": "beige",
        "color_family": "neutral",
        "pattern": "plain",
        "material": "cotton",
        "fit_type": "slim",
        "silhouette": "tapered",
        "length": "full_length",
        "layer": "base",
        "formality": "smart_casual",
        "occasions": ["work_formal", "casual_outing", "college", "date_night"],
        "seasons": ["summer", "spring", "autumn"],
        "style_tags": ["smart_casual", "minimal", "classic"],
    },
    {
        "slug": "olive_tailored_chinos",
        "image_url": "/static/predefined_wardrobe/olive_tailored_chinos.jpg",
        "name": "Olive Khaki Tailored Stretch Chinos",
        "category": "bottom",
        "subcategory": "chinos",
        "primary_color": "green",
        "shade": "olive_green",
        "color_family": "earthy",
        "pattern": "plain",
        "material": "cotton",
        "fit_type": "slim",
        "silhouette": "tapered",
        "length": "full_length",
        "layer": "base",
        "formality": "smart_casual",
        "occasions": ["work_formal", "casual_outing", "college", "date_night"],
        "seasons": ["summer", "spring", "autumn"],
        "style_tags": ["smart_casual", "casual", "classic"],
    },
    {
        "slug": "indigo_straight_jeans",
        "image_url": "/static/predefined_wardrobe/indigo_straight_jeans.jpg",
        "name": "Classic Indigo Slim-Straight Denim Jeans",
        "category": "bottom",
        "subcategory": "jeans",
        "primary_color": "blue",
        "shade": "indigo_denim",
        "color_family": "cool",
        "pattern": "plain",
        "material": "denim",
        "fit_type": "regular",
        "silhouette": "straight",
        "length": "full_length",
        "layer": "base",
        "formality": "casual",
        "occasions": ["casual_outing", "college", "date_night", "party"],
        "seasons": ["summer", "spring", "autumn", "winter"],
        "style_tags": ["casual", "streetwear", "classic"],
    },
    {
        "slug": "black_denim_jeans",
        "image_url": "/static/predefined_wardrobe/black_denim_jeans.jpg",
        "name": "Washed Black Slim-Straight Denim Jeans",
        "category": "bottom",
        "subcategory": "jeans",
        "primary_color": "black",
        "shade": "black",
        "color_family": "neutral",
        "pattern": "plain",
        "material": "denim",
        "fit_type": "slim",
        "silhouette": "straight",
        "length": "full_length",
        "layer": "base",
        "formality": "casual",
        "occasions": ["casual_outing", "college", "date_night", "party"],
        "seasons": ["summer", "spring", "autumn", "winter"],
        "style_tags": ["streetwear", "casual", "edgy"],
    },
    # --- MEN'S GYM JOGGERS & TRACK PANTS (3) ---
    {
        "slug": "navy_cotton_joggers",
        "image_url": "/static/predefined_wardrobe/navy_cotton_joggers.jpg",
        "name": "Dark Navy Tapered Athletic Gym Joggers",
        "category": "bottom",
        "subcategory": "joggers",
        "primary_color": "navy",
        "shade": "dark_navy",
        "color_family": "neutral",
        "pattern": "plain",
        "material": "cotton",
        "fit_type": "regular",
        "silhouette": "tapered",
        "length": "full_length",
        "layer": "base",
        "formality": "casual",
        "occasions": ["gym_sports", "casual_outing", "college"],
        "seasons": ["summer", "spring", "autumn", "winter"],
        "style_tags": ["sporty", "athletic", "gymwear", "streetwear"],
    },
    {
        "slug": "charcoal_gym_track_pants",
        "image_url": "/static/predefined_wardrobe/charcoal_gym_track_pants.jpg",
        "name": "Charcoal Grey Performance Gym Track Pants",
        "category": "bottom",
        "subcategory": "joggers",
        "primary_color": "grey",
        "shade": "charcoal_grey",
        "color_family": "neutral",
        "pattern": "plain",
        "material": "polyester",
        "fit_type": "regular",
        "silhouette": "tapered",
        "length": "full_length",
        "layer": "base",
        "formality": "casual",
        "occasions": ["gym_sports"],
        "seasons": ["summer", "spring", "autumn", "winter"],
        "style_tags": ["sporty", "athletic", "gymwear"],
    },
    {
        "slug": "black_athletic_joggers",
        "image_url": "/static/predefined_wardrobe/black_athletic_joggers.jpg",
        "name": "Jet Black Dry-Fit Athletic Gym Joggers",
        "category": "bottom",
        "subcategory": "joggers",
        "primary_color": "black",
        "shade": "black",
        "color_family": "neutral",
        "pattern": "plain",
        "material": "polyester",
        "fit_type": "slim",
        "silhouette": "tapered",
        "length": "full_length",
        "layer": "base",
        "formality": "casual",
        "occasions": ["gym_sports"],
        "seasons": ["summer", "spring", "autumn", "winter"],
        "style_tags": ["sporty", "athletic", "gymwear", "streetwear"],
    },
    # --- MEN'S WINTER COATS, SUIT BLAZERS & JACKETS (9) ---
    {
        "slug": "camel_wool_overcoat",
        "image_url": "/static/predefined_wardrobe/camel_wool_overcoat.jpg",
        "name": "Camel Tan Wool Tailored Winter Overcoat",
        "category": "overtop",
        "subcategory": "coat",
        "primary_color": "tan",
        "shade": "tan",
        "color_family": "neutral",
        "pattern": "plain",
        "material": "wool",
        "fit_type": "tailored",
        "silhouette": "structured",
        "length": "long",
        "layer": "outer",
        "formality": "formal",
        "occasions": ["work_formal", "date_night", "party", "wedding"],
        "seasons": ["winter", "autumn"],
        "style_tags": ["formal", "classic", "chic"],
    },
    {
        "slug": "charcoal_wool_overcoat",
        "image_url": "/static/predefined_wardrobe/charcoal_wool_overcoat.jpg",
        "name": "Charcoal Black Wool Single-Breasted Overcoat",
        "category": "overtop",
        "subcategory": "coat",
        "primary_color": "grey",
        "shade": "charcoal_grey",
        "color_family": "neutral",
        "pattern": "plain",
        "material": "wool",
        "fit_type": "tailored",
        "silhouette": "structured",
        "length": "long",
        "layer": "outer",
        "formality": "formal",
        "occasions": ["work_formal", "date_night", "party", "wedding"],
        "seasons": ["winter", "autumn"],
        "style_tags": ["formal", "classic", "minimal"],
    },
    {
        "slug": "navy_tailored_suit_coat",
        "image_url": "/static/predefined_wardrobe/navy_tailored_suit_coat.jpg",
        "name": "Midnight Navy Tailored Wool Suit Blazer",
        "category": "overtop",
        "subcategory": "blazer",
        "primary_color": "navy",
        "shade": "navy",
        "color_family": "neutral",
        "pattern": "plain",
        "material": "wool",
        "fit_type": "tailored",
        "silhouette": "structured",
        "length": "regular",
        "layer": "outer",
        "formality": "formal",
        "occasions": ["work_formal", "date_night", "wedding", "party"],
        "seasons": ["winter", "autumn", "spring"],
        "style_tags": ["formal", "classic", "chic"],
    },
    {
        "slug": "black_leather_biker_jacket",
        "image_url": "/static/predefined_wardrobe/black_leather_biker_jacket.jpg",
        "name": "Obsidian Black Leather Moto Biker Jacket",
        "category": "overtop",
        "subcategory": "jacket",
        "primary_color": "black",
        "shade": "black",
        "color_family": "neutral",
        "pattern": "plain",
        "material": "leather",
        "fit_type": "regular",
        "silhouette": "structured",
        "length": "regular",
        "layer": "outer",
        "formality": "casual",
        "occasions": ["casual_outing", "college", "party", "date_night"],
        "seasons": ["winter", "autumn"],
        "style_tags": ["streetwear", "casual", "edgy"],
    },
    {
        "slug": "brown_leather_flight_jacket",
        "image_url": "/static/predefined_wardrobe/brown_leather_flight_jacket.jpg",
        "name": "Cognac Brown Leather Moto Flight Jacket",
        "category": "overtop",
        "subcategory": "jacket",
        "primary_color": "brown",
        "shade": "chocolate_brown",
        "color_family": "earthy",
        "pattern": "plain",
        "material": "leather",
        "fit_type": "regular",
        "silhouette": "structured",
        "length": "regular",
        "layer": "outer",
        "formality": "smart_casual",
        "occasions": ["casual_outing", "college", "date_night", "party"],
        "seasons": ["winter", "autumn"],
        "style_tags": ["streetwear", "casual", "classic"],
    },
    {
        "slug": "navy_winter_puffer_jacket",
        "image_url": "/static/predefined_wardrobe/navy_winter_puffer_jacket.jpg",
        "name": "Dark Navy Quilted Down Winter Puffer Jacket",
        "category": "overtop",
        "subcategory": "puffer_jacket",
        "primary_color": "navy",
        "shade": "dark_navy",
        "color_family": "neutral",
        "pattern": "plain",
        "material": "polyester",
        "fit_type": "regular",
        "silhouette": "structured",
        "length": "regular",
        "layer": "outer",
        "formality": "casual",
        "occasions": ["casual_outing", "college", "vacation", "gym_sports"],
        "seasons": ["winter"],
        "style_tags": ["casual", "streetwear", "winter", "sporty"],
    },
    {
        "slug": "olive_winter_puffer_jacket",
        "image_url": "/static/predefined_wardrobe/olive_winter_puffer_jacket.jpg",
        "name": "Olive Green Quilted Down Winter Puffer Jacket",
        "category": "overtop",
        "subcategory": "puffer_jacket",
        "primary_color": "green",
        "shade": "olive_green",
        "color_family": "earthy",
        "pattern": "plain",
        "material": "polyester",
        "fit_type": "regular",
        "silhouette": "structured",
        "length": "regular",
        "layer": "outer",
        "formality": "casual",
        "occasions": ["casual_outing", "college", "vacation", "gym_sports"],
        "seasons": ["winter", "autumn"],
        "style_tags": ["casual", "streetwear", "winter"],
    },
    {
        "slug": "white_winter_bomber_jacket",
        "image_url": "/static/predefined_wardrobe/white_winter_bomber_jacket.jpg",
        "name": "Off-White Ribbed Winter Zip Bomber Jacket",
        "category": "overtop",
        "subcategory": "bomber_jacket",
        "primary_color": "white",
        "shade": "off_white",
        "color_family": "neutral",
        "pattern": "plain",
        "material": "polyester",
        "fit_type": "regular",
        "silhouette": "structured",
        "length": "regular",
        "layer": "outer",
        "formality": "casual",
        "occasions": ["casual_outing", "college", "party", "gym_sports"],
        "seasons": ["winter", "autumn"],
        "style_tags": ["streetwear", "casual", "minimal", "sporty"],
    },
    {
        "slug": "black_varsity_bomber_jacket",
        "image_url": "/static/predefined_wardrobe/black_varsity_bomber_jacket.jpg",
        "name": "Matte Black Ribbed Streetwear Bomber Jacket",
        "category": "overtop",
        "subcategory": "bomber_jacket",
        "primary_color": "black",
        "shade": "black",
        "color_family": "neutral",
        "pattern": "plain",
        "material": "polyester",
        "fit_type": "regular",
        "silhouette": "structured",
        "length": "regular",
        "layer": "outer",
        "formality": "casual",
        "occasions": ["casual_outing", "college", "party", "gym_sports"],
        "seasons": ["winter", "autumn", "spring"],
        "style_tags": ["streetwear", "casual", "sporty"],
    },
    # --- MEN'S SHOES, SNEAKERS & GYM TRAINERS (8) ---
    {
        "slug": "white_leather_sneakers",
        "image_url": "/static/predefined_wardrobe/white_leather_sneakers.jpg",
        "name": "Minimalist Pure White Low-Top Leather Sneakers",
        "category": "footwear",
        "subcategory": "sneakers",
        "primary_color": "white",
        "shade": "pure_white",
        "color_family": "neutral",
        "pattern": "plain",
        "material": "leather",
        "fit_type": "regular",
        "silhouette": "straight",
        "length": "regular",
        "layer": "base",
        "formality": "casual",
        "occasions": ["casual_outing", "college", "date_night", "party"],
        "seasons": ["summer", "spring", "autumn", "winter"],
        "style_tags": ["casual", "streetwear", "minimal"],
    },
    {
        "slug": "olive_suede_sneakers",
        "image_url": "/static/predefined_wardrobe/olive_suede_sneakers.jpg",
        "name": "Olive & Grey Suede Streetwear Runner Sneakers",
        "category": "footwear",
        "subcategory": "sneakers",
        "primary_color": "green",
        "shade": "olive_green",
        "color_family": "earthy",
        "pattern": "colourblock",
        "material": "leather",
        "fit_type": "regular",
        "silhouette": "structured",
        "length": "regular",
        "layer": "base",
        "formality": "casual",
        "occasions": ["casual_outing", "college", "gym_sports"],
        "seasons": ["summer", "spring", "autumn", "winter"],
        "style_tags": ["streetwear", "casual", "sporty"],
    },
    {
        "slug": "black_gym_running_shoes",
        "image_url": "/static/predefined_wardrobe/black_gym_running_shoes.jpg",
        "name": "Stealth Black Athletic Gym Running Shoes",
        "category": "footwear",
        "subcategory": "sports_shoes",
        "primary_color": "black",
        "shade": "black",
        "color_family": "neutral",
        "pattern": "plain",
        "material": "polyester",
        "fit_type": "regular",
        "silhouette": "structured",
        "length": "regular",
        "layer": "base",
        "formality": "casual",
        "occasions": ["gym_sports"],
        "seasons": ["summer", "spring", "autumn", "winter"],
        "style_tags": ["sporty", "athletic", "gymwear"],
    },
    {
        "slug": "cobalt_running_trainers",
        "image_url": "/static/predefined_wardrobe/cobalt_running_trainers.jpg",
        "name": "Navy & Grey Mesh Gym Running Trainers",
        "category": "footwear",
        "subcategory": "sports_shoes",
        "primary_color": "navy",
        "shade": "navy",
        "color_family": "cool",
        "pattern": "colourblock",
        "material": "polyester",
        "fit_type": "regular",
        "silhouette": "structured",
        "length": "regular",
        "layer": "base",
        "formality": "casual",
        "occasions": ["gym_sports"],
        "seasons": ["summer", "spring", "autumn", "winter"],
        "style_tags": ["sporty", "athletic", "gymwear"],
    },
    {
        "slug": "tan_penny_loafers",
        "image_url": "/static/predefined_wardrobe/tan_penny_loafers.jpg",
        "name": "Tan Brown Suede Classic Penny Loafers",
        "category": "footwear",
        "subcategory": "loafers",
        "primary_color": "tan",
        "shade": "tan",
        "color_family": "neutral",
        "pattern": "plain",
        "material": "leather",
        "fit_type": "regular",
        "silhouette": "structured",
        "length": "regular",
        "layer": "base",
        "formality": "smart_casual",
        "occasions": ["work_formal", "date_night", "casual_outing", "festival", "wedding", "party"],
        "seasons": ["summer", "spring", "autumn", "winter"],
        "style_tags": ["smart_casual", "classic", "formal", "indo_western"],
    },
    {
        "slug": "black_suede_loafers",
        "image_url": "/static/predefined_wardrobe/black_suede_loafers.jpg",
        "name": "Jet Black Suede Dress Penny Loafers",
        "category": "footwear",
        "subcategory": "loafers",
        "primary_color": "black",
        "shade": "black",
        "color_family": "neutral",
        "pattern": "plain",
        "material": "leather",
        "fit_type": "regular",
        "silhouette": "structured",
        "length": "regular",
        "layer": "base",
        "formality": "semi_formal",
        "occasions": ["work_formal", "date_night", "festival", "wedding", "party"],
        "seasons": ["summer", "spring", "autumn", "winter"],
        "style_tags": ["smart_casual", "classic", "formal", "ethnic"],
    },
    {
        "slug": "black_captoe_oxfords",
        "image_url": "/static/predefined_wardrobe/black_captoe_oxfords.jpg",
        "name": "Black Polished Leather Cap-Toe Oxford Shoes",
        "category": "footwear",
        "subcategory": "oxford",
        "primary_color": "black",
        "shade": "black",
        "color_family": "neutral",
        "pattern": "plain",
        "material": "leather",
        "fit_type": "tailored",
        "silhouette": "structured",
        "length": "regular",
        "layer": "base",
        "formality": "formal",
        "occasions": ["work_formal", "wedding", "date_night"],
        "seasons": ["summer", "spring", "autumn", "winter"],
        "style_tags": ["formal", "classic"],
    },
    {
        "slug": "brown_leather_oxfords",
        "image_url": "/static/predefined_wardrobe/brown_leather_oxfords.jpg",
        "name": "Espresso Brown Leather Cap-Toe Oxford Shoes",
        "category": "footwear",
        "subcategory": "oxford",
        "primary_color": "brown",
        "shade": "chocolate_brown",
        "color_family": "earthy",
        "pattern": "plain",
        "material": "leather",
        "fit_type": "tailored",
        "silhouette": "structured",
        "length": "regular",
        "layer": "base",
        "formality": "formal",
        "occasions": ["work_formal", "wedding", "date_night"],
        "seasons": ["summer", "spring", "autumn", "winter"],
        "style_tags": ["formal", "classic"],
    },
]


def init_db(seed_if_empty: bool = False):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS user_profile (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            name TEXT NOT NULL DEFAULT 'Tanishk',
            dob TEXT DEFAULT '2003-05-15',
            city TEXT DEFAULT 'Mumbai',
            latitude REAL DEFAULT 19.0760,
            longitude REAL DEFAULT 72.8777,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cur.execute("SELECT COUNT(*) as cnt FROM user_profile WHERE id = 1")
    if cur.fetchone()["cnt"] == 0:
        cur.execute(
            """
            INSERT INTO user_profile (id, name, dob, city, latitude, longitude)
            VALUES (1, 'Tanishk', '2003-05-15', 'Mumbai', 19.0760, 72.8777)
            """
        )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS wardrobe_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            label TEXT,
            category TEXT NOT NULL,
            subcategory TEXT NOT NULL,
            primary_color TEXT NOT NULL,
            shade TEXT NOT NULL,
            secondary_color TEXT,
            color_family TEXT DEFAULT 'neutral',
            pattern TEXT DEFAULT 'plain',
            material TEXT DEFAULT 'cotton',
            fit_type TEXT DEFAULT 'regular',
            silhouette TEXT DEFAULT 'straight',
            length TEXT DEFAULT 'regular',
            layer TEXT DEFAULT 'base',
            formality TEXT DEFAULT 'casual',
            occasions_json TEXT DEFAULT '[]',
            seasons_json TEXT DEFAULT '[]',
            weather_json TEXT DEFAULT '[]',
            style_tags_json TEXT DEFAULT '[]',
            palette_json TEXT DEFAULT '[]',
            image_url TEXT,
            mask_url TEXT,
            ghost_mannequin_url TEXT,
            ghost_transparent_url TEXT,
            ghost_source TEXT,
            visual_signature TEXT,
            full_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.commit()
    cur.execute("SELECT COUNT(*) as cnt FROM wardrobe_items")
    count = cur.fetchone()["cnt"]
    conn.close()

    if seed_if_empty and count == 0:
        seed_predefined_wardrobe()


def compute_visual_signature(item_data: Dict[str, Any]) -> List[float]:
    """
    Computes a compact 24-d normalized visual + color-histogram signature from the
    garment image file and its CIELAB palette for accurate duplicate detection.
    """
    vec = np.zeros(24, dtype=np.float32)

    raw_color = item_data.get("color", {})
    color_obj = raw_color if isinstance(raw_color, dict) else {}
    palette = color_obj.get("palette", [])
    for idx, p in enumerate(palette[:3]):
        hex_str = p.get("hex", "#FFFFFF").lstrip("#")
        if len(hex_str) == 6:
            r = int(hex_str[0:2], 16) / 255.0
            g = int(hex_str[2:4], 16) / 255.0
            b = int(hex_str[4:6], 16) / 255.0
            pct = float(p.get("percent", 0.0)) / 100.0
            base = idx * 4
            vec[base : base + 4] = [r, g, b, pct]

    img_url = item_data.get("image_url") or item_data.get("ghost_mannequin_url") or ""
    if img_url.startswith("/static/"):
        local_path = STATIC_DIR / img_url.replace("/static/", "")
        if local_path.exists():
            img = cv2.imread(str(local_path), cv2.IMREAD_UNCHANGED)
            if img is not None and img.size > 0:
                if img.shape[-1] == 4:
                    alpha = img[:, :, 3] > 20
                    bgr = img[:, :, :3]
                    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                    valid = hsv[alpha]
                else:
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    valid = hsv.reshape(-1, 3)
                if len(valid) > 20:
                    h_hist, _ = np.histogram(valid[:, 0], bins=6, range=(0, 180), density=True)
                    s_hist, _ = np.histogram(valid[:, 1], bins=3, range=(0, 256), density=True)
                    v_hist, _ = np.histogram(valid[:, 2], bins=3, range=(0, 256), density=True)
                    feat12 = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
                    norm12 = np.linalg.norm(feat12)
                    if norm12 > 1e-6:
                        vec[12:24] = feat12 / norm12

    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        vec = vec / norm
    return [round(float(x), 5) for x in vec]


def cosine_sim(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def get_profile() -> Dict[str, Any]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM user_profile WHERE id = 1").fetchone()
    conn.close()
    if not row:
        return {
            "name": "Tanishk",
            "dob": "2003-05-15",
            "city": "Mumbai",
            "latitude": 19.0760,
            "longitude": 72.8777,
        }
    return dict(row)


def update_profile(
    name: Optional[str] = None,
    dob: Optional[str] = None,
    city: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> Dict[str, Any]:
    current = get_profile()
    new_name = name if name is not None else current["name"]
    new_dob = dob if dob is not None else current["dob"]
    new_city = city if city is not None else current["city"]
    new_lat = latitude if latitude is not None else current["latitude"]
    new_lon = longitude if longitude is not None else current["longitude"]

    conn = get_conn()
    conn.execute(
        """
        UPDATE user_profile
        SET name = ?, dob = ?, city = ?, latitude = ?, longitude = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = 1
        """,
        (new_name, new_dob, new_city, new_lat, new_lon),
    )
    conn.commit()
    conn.close()
    return get_profile()


def _row_to_item(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    return {
        "id": d["id"],
        "name": d["name"],
        "label": d["label"] or d["name"],
        "category": d["category"],
        "subcategory": d["subcategory"],
        "primary_color": d["primary_color"],
        "shade": d["shade"],
        "material": d["material"],
        "pattern": d["pattern"],
        "fit_type": d["fit_type"],
        "silhouette": d["silhouette"],
        "length": d["length"],
        "layer": d["layer"],
        "formality": d["formality"],
        "image_url": d["image_url"],
        "mask_url": d["mask_url"],
        "ghost_mannequin_url": d["ghost_mannequin_url"] or d["image_url"],
        "ghost_transparent_url": d["ghost_transparent_url"],
        "ghost_source": d["ghost_source"] or "ghost_cropped",
        "color": {
            "primary": d["primary_color"],
            "shade": d["shade"],
            "secondary": d["secondary_color"],
            "family": d["color_family"],
            "palette": json.loads(d["palette_json"] or "[]"),
            "pattern": d["pattern"],
        },
        "fit": {
            "type": d["fit_type"],
            "silhouette": d["silhouette"],
            "length": d["length"],
        },
        "occasion": {
            "types": json.loads(d["occasions_json"] or "[]"),
            "formality": d["formality"],
        },
        "weather": {
            "seasons": json.loads(d["seasons_json"] or "[]"),
            "conditions": json.loads(d["weather_json"] or "[]"),
            "material": d["material"],
            "layering": d["layer"],
        },
        "style_tags": json.loads(d["style_tags_json"] or "[]"),
        "created_at": d["created_at"],
    }


def list_wardrobe_items(category: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_conn()
    if category and category != "all":
        rows = conn.execute(
            "SELECT * FROM wardrobe_items WHERE category = ? ORDER BY id DESC",
            (category.lower(),),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM wardrobe_items ORDER BY id DESC").fetchall()
    conn.close()
    return [_row_to_item(r) for r in rows]


def add_item_to_wardrobe(item_data: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, str]:
    """
    Adds a detected garment item to the user's wardrobe.
    Checks if the same item is already present in the wardrobe:
    - Exact match on (category, subcategory, shade, pattern, material, fit_type)
    - OR same (category, subcategory) with visual signature cosine similarity >= 0.96
      and matching shade & pattern.
    Returns (item_dict, is_duplicate, message).
    """
    category = str(item_data.get("category") or "top").lower()
    subcategory = str(item_data.get("subcategory") or "tshirt").lower()
    raw_color = item_data.get("color", {})
    color_obj = raw_color if isinstance(raw_color, dict) else {}
    raw_fit = item_data.get("fit", {})
    fit_obj = raw_fit if isinstance(raw_fit, dict) else {}
    raw_weather = item_data.get("weather", {})
    weather_obj = raw_weather if isinstance(raw_weather, dict) else {}
    raw_occasion = item_data.get("occasion", {})
    occasion_obj = raw_occasion if isinstance(raw_occasion, dict) else {}

    primary_color = str(color_obj.get("primary") or item_data.get("primary_color") or "white").lower()
    shade = str(color_obj.get("shade") or item_data.get("shade") or primary_color).lower()
    secondary_color = color_obj.get("secondary") or item_data.get("secondary_color")
    color_family = str(color_obj.get("family") or item_data.get("color_family") or "neutral").lower()
    pattern = str(item_data.get("pattern") or color_obj.get("pattern") or "plain").lower()
    material = str(item_data.get("material") or weather_obj.get("material") or "cotton").lower()
    fit_str = raw_fit if isinstance(raw_fit, str) else None
    fit_type = str(item_data.get("fit_type") or fit_str or fit_obj.get("type") or "regular").lower()
    silhouette = str(item_data.get("silhouette") or fit_obj.get("silhouette") or "straight").lower()
    length = str(item_data.get("length") or fit_obj.get("length") or "regular").lower()
    layer = str(item_data.get("layer") or item_data.get("layering") or weather_obj.get("layering") or "base").lower()
    formality = str(item_data.get("formality") or occasion_obj.get("formality") or "casual").lower()

    occasions = occasion_obj.get("types") or item_data.get("occasions") or ["casual_outing"]
    seasons = weather_obj.get("seasons") or item_data.get("seasons") or ([item_data["season"]] if item_data.get("season") else ["summer", "spring"])
    weather_conds = weather_obj.get("conditions") or item_data.get("weather_conditions") or ["warm"]
    style_tags = item_data.get("style_tags") or ["casual"]
    palette = color_obj.get("palette") or [
        {"name": shade.replace("_", " ").title(), "shade": shade, "hex": SHADE_HEX_MAP.get(shade, "#FFFFFF"), "percent": 100.0}
    ]

    new_sig = compute_visual_signature(item_data)

    conn = get_conn()
    existing_rows = conn.execute(
        "SELECT * FROM wardrobe_items WHERE category = ? AND subcategory = ?",
        (category, subcategory),
    ).fetchall()

    for row in existing_rows:
        ex_shade = (row["shade"] or "").lower()
        ex_pattern = (row["pattern"] or "").lower()
        ex_material = (row["material"] or "").lower()
        ex_fit = (row["fit_type"] or "").lower()
        ex_sig = json.loads(row["visual_signature"] or "[]")
        sim = cosine_sim(new_sig, ex_sig)

        # Duplicate Condition 1: Identical subcategory + shade + pattern + material + fit_type
        if ex_shade == shade and ex_pattern == pattern and ex_material == material and ex_fit == fit_type:
            existing_item = _row_to_item(row)
            conn.close()
            return (
                existing_item,
                True,
                f"Already in wardrobe: '{existing_item['name']}' ({subcategory.replace('_', ' ')}, {shade.replace('_', ' ')}) is already saved.",
            )

        # Duplicate Condition 2: Same subcategory + same shade + same pattern + high visual similarity (>= 0.96)
        if ex_shade == shade and ex_pattern == pattern and sim >= 0.96:
            existing_item = _row_to_item(row)
            conn.close()
            return (
                existing_item,
                True,
                f"Already in wardrobe: '{existing_item['name']}' matched with {sim*100:.0f}% visual similarity.",
            )

    name = item_data.get("name") or f"{shade.replace('_', ' ').title()} {subcategory.replace('_', ' ').title()}"
    label = item_data.get("label") or name
    img_url = item_data.get("image_url", "")

    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO wardrobe_items (
            name, label, category, subcategory, primary_color, shade, secondary_color,
            color_family, pattern, material, fit_type, silhouette, length, layer,
            formality, occasions_json, seasons_json, weather_json, style_tags_json,
            palette_json, image_url, mask_url, ghost_mannequin_url, ghost_transparent_url,
            ghost_source, visual_signature, full_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            name,
            label,
            category,
            subcategory,
            primary_color,
            shade,
            secondary_color,
            color_family,
            pattern,
            material,
            fit_type,
            silhouette,
            length,
            layer,
            formality,
            json.dumps(occasions),
            json.dumps(seasons),
            json.dumps(weather_conds),
            json.dumps(style_tags),
            json.dumps(palette),
            img_url,
            item_data.get("mask_url", ""),
            item_data.get("ghost_mannequin_url", img_url),
            item_data.get("ghost_transparent_url", img_url),
            item_data.get("ghost_source", "nano_banana_studio"),
            json.dumps(new_sig),
            json.dumps(item_data),
        ),
    )
    new_id = cur.lastrowid
    conn.commit()
    new_row = conn.execute("SELECT * FROM wardrobe_items WHERE id = ?", (new_id,)).fetchone()
    conn.close()

    return _row_to_item(new_row), False, f"Added '{name}' to your wardrobe."


def seed_predefined_wardrobe() -> Dict[str, Any]:
    """
    Populates the wardrobe with the 30 photorealistic Nano Banana 3D ghost-mannequin
    studio garments & footwear in static/predefined_wardrobe/.
    """
    added_count = 0
    skipped_count = 0
    for spec in PREDEFINED_WARDROBE_CATALOG:
        payload = dict(spec)
        img_url = spec["image_url"]
        payload["ghost_mannequin_url"] = img_url
        payload["ghost_transparent_url"] = img_url
        payload["ghost_source"] = "nano_banana_studio"
        _, is_dup, _ = add_item_to_wardrobe(payload)
        if is_dup:
            skipped_count += 1
        else:
            added_count += 1

    return {
        "added_count": added_count,
        "skipped_count": skipped_count,
        "total_wardrobe": len(list_wardrobe_items()),
    }


def delete_wardrobe_item(item_id: int) -> bool:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM wardrobe_items WHERE id = ?", (item_id,))
    deleted = cur.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def clear_wardrobe():
    conn = get_conn()
    conn.execute("DELETE FROM wardrobe_items")
    conn.commit()
    conn.close()


init_db(seed_if_empty=True)
