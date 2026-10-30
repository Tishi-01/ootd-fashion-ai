# taxonomy.py
"""
Unified Fashion Taxonomy for OOTD Garment Classification & Metadata Extraction.
Includes Western and Indian Ethnic / Fusion wear, Outerwear, Footwear, and Accessories.
"""

CATEGORIES = ["top", "bottom", "overtop", "dress", "footwear", "accessory"]

SUBCATEGORIES = {
    "top": [
        "tshirt", "shirt", "polo", "tank_top", "crop_top", "blouse", "corset",
        "sweatshirt", "sweater", "kurti", "kurta", "saree_blouse"
    ],
    "bottom": [
        "jeans", "trousers", "wide_leg_trousers", "formal_trousers", "chinos", "joggers", "track_pants",
        "shorts", "skirt", "cargo_pants", "sweatpants", "leggings",
        "palazzo", "salwar", "churidar", "dhoti_pants"
    ],
    "overtop": [
        "blazer", "bomber_jacket", "denim_jacket", "leather_jacket", "puffer_jacket", "jacket",
        "cardigan", "hoodie", "trench_coat", "vest",
        "dupatta_stole", "nehru_jacket", "shawl"
    ],
    "dress": [
        "casual_dress", "maxi_dress", "slip_dress", "bodycon_dress", "shirt_dress",
        "saree", "anarkali", "lehenga"
    ],
    "footwear": [
        "sneakers", "boots", "sandals", "platform_sandals", "formal_shoes", "loafers", "heels", "flats",
        "juttis_mojaris", "kolhapuris", "ethnic_flats"
    ],
    "accessory": [
        "handbag", "backpack", "belt", "watch", "bracelet", "sunglasses", "cap", "scarf"
    ]
}

# Flattened unique subcategories
ALL_SUBCATEGORIES = sorted(list(set([sub for subs in SUBCATEGORIES.values() for sub in subs])))

# Map each subcategory to its valid parent category
SUBCAT_TO_PARENT = {}
for parent_cat, sub_list in SUBCATEGORIES.items():
    for sub in sub_list:
        SUBCAT_TO_PARENT[sub] = parent_cat

PATTERNS = ["plain", "striped", "checkered", "graphic", "floral", "polka_dot", "distressed", "colorblock", "tie_dye"]

FIT_TYPES = ["oversized", "slim", "regular", "skinny", "baggy", "tailored", "loose"]
SILHOUETTES = ["relaxed", "fitted", "boxy", "straight", "flared", "a_line", "tapered"]
LENGTHS = ["crop", "regular", "longline", "knee_length", "ankle_length", "mini", "maxi"]

OCCASIONS = [
    "college", "casual_outing", "work_formal", "party", "date_night",
    "gym_sports", "lounge", "vacation", "festival", "wedding"
]
FORMALITIES = ["casual", "smart_casual", "semi_formal", "formal"]

SEASONS = ["spring", "summer", "fall", "winter"]
CONDITIONS = ["hot", "warm", "cool", "cold", "rainy", "windy"]
MATERIALS = [
    "cotton", "denim", "polyester", "linen", "leather", "wool", "silk",
    "nylon", "knit", "corduroy", "chiffon_georgette", "velvet", "tweed"
]
LAYERING_TYPES = ["none", "base", "inner", "mid", "outer"]

STYLE_TAGS = [
    "casual", "streetwear", "minimal", "vintage", "y2k", "sporty",
    "formal", "chic", "grunge", "preppy", "boho", "athleisure",
    "ethnic", "indo_western"
]

# Mapping helpers for quick index lookup
CAT2IDX = {c: i for i, c in enumerate(CATEGORIES)}
SUBCAT2IDX = {c: i for i, c in enumerate(ALL_SUBCATEGORIES)}
PATTERN2IDX = {c: i for i, c in enumerate(PATTERNS)}
FIT2IDX = {c: i for i, c in enumerate(FIT_TYPES)}
SILHOUETTE2IDX = {c: i for i, c in enumerate(SILHOUETTES)}
LENGTH2IDX = {c: i for i, c in enumerate(LENGTHS)}
FORMALITY2IDX = {c: i for i, c in enumerate(FORMALITIES)}
MATERIAL2IDX = {c: i for i, c in enumerate(MATERIALS)}
LAYERING2IDX = {c: i for i, c in enumerate(LAYERING_TYPES)}

OCCASION2IDX = {c: i for i, c in enumerate(OCCASIONS)}
SEASON2IDX = {c: i for i, c in enumerate(SEASONS)}
CONDITION2IDX = {c: i for i, c in enumerate(CONDITIONS)}
STYLE2IDX = {c: i for i, c in enumerate(STYLE_TAGS)}
