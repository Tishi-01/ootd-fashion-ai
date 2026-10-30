"""
Deterministic Rules Module for OOTD Fashion AI.
Replaces noisy subjective neural heads with clean, expert-curated fashion rules.
Derives formality, occasions, seasons, weather suitability, and style tags
from recognized garment attributes (subcategory, material, fit, color).
"""

import os
import yaml
from pathlib import Path

DEFAULT_RULES_PATH = Path(__file__).resolve().parent / "rules.yaml"

class FashionRulesEngine:
    _instance = None

    def __init__(self, rules_path=None):
        self.rules_path = Path(rules_path or DEFAULT_RULES_PATH)
        self.rules = self._load_rules()

    @classmethod
    def get_instance(cls, rules_path=None):
        if cls._instance is None:
            cls._instance = cls(rules_path)
        return cls._instance

    def _load_rules(self):
        if not self.rules_path.exists():
            return {"subcategories": {}, "materials_weather_modifier": {}}
        with open(self.rules_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def derive_attributes(self, category, subcategory, material="cotton", fit="regular", color="white", shade="pure_white"):
        """
        Derives subjective and contextual attributes deterministically.
        """
        subcat_key = (subcategory or "").strip().lower()
        mat_key = (material or "cotton").strip().lower()
        col_key = (color or "neutral").strip().lower()

        subcat_rule = self.rules.get("subcategories", {}).get(subcat_key, {})

        # 1. Formality
        base_formality = subcat_rule.get("formality", "casual")
        modifiers = subcat_rule.get("formality_modifiers", {})
        if mat_key in modifiers:
            base_formality = modifiers[mat_key]
        elif col_key in modifiers:
            base_formality = modifiers[col_key]

        # 2. Occasions
        occasions = list(subcat_rule.get("occasions", ["casual_outing"]))
        # Color & material occasion modifiers
        if mat_key in ["silk", "velvet"] and "wedding" not in occasions:
            occasions.append("wedding")
        if mat_key in ["cotton", "linen"] and "summer" in subcat_rule.get("seasons", []):
            if "casual_outing" not in occasions:
                occasions.append("casual_outing")

        # 3. Seasons & Weather
        seasons = list(subcat_rule.get("seasons", ["spring", "summer", "fall"]))
        weather = list(subcat_rule.get("weather", ["warm"]))

        mat_weather = self.rules.get("materials_weather_modifier", {}).get(mat_key, [])
        for w in mat_weather:
            if w not in weather:
                weather.append(w)

        # 4. Style Tags
        style_tags = list(subcat_rule.get("style_tags", ["casual"]))
        if "ethnic" in style_tags and base_formality in ["casual", "smart_casual"]:
            if "indo_western" not in style_tags:
                style_tags.append("indo_western")

        # 5. Layering Role
        layering = subcat_rule.get("layering")
        if not layering:
            cat_lower = (category or "").lower()
            if cat_lower == "overtop" or subcat_key in ["dupatta_stole", "shawl", "nehru_jacket", "blazer", "jacket", "bomber_jacket", "leather_jacket", "puffer_jacket"]:
                layering = "outer"
            elif cat_lower == "top" and subcat_key in ["sweater", "cardigan", "hoodie", "sweatshirt"]:
                layering = "mid"
            elif cat_lower in ["top", "bottom"]:
                layering = "base"
            else:
                layering = "none"

        return {
            "formality": base_formality,
            "occasions": sorted(list(set(occasions))),
            "seasons": sorted(list(set(seasons))),
            "weather": sorted(list(set(weather))),
            "style_tags": sorted(list(set(style_tags))),
            "layering": layering
        }

def derive_garment_metadata(category, subcategory, material="cotton", fit="regular", color="white", shade="pure_white", config=None):
    engine = FashionRulesEngine.get_instance()
    derived = engine.derive_attributes(category, subcategory, material, fit, color, shade)

    # Optional LLM Enrichment if configured and API key present
    if config and config.get("rules", {}).get("enable_llm_enrichment", False):
        # Fallback to deterministic if offline or disabled
        pass

    return derived
