"""Shared constants for the Diamond Analytics dashboard.

Centralises pitch type colors and labels so every chart in the dashboard
uses a consistent visual language.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Canonical pitch-type color map
# ---------------------------------------------------------------------------
# Grouped by pitch family so similar pitches sit in the same hue band:
#   Fastball family  -> reds / oranges
#   Breaking balls   -> blues
#   Curveball family -> greens
#   Off-speed        -> purples
#   Novelty          -> neutrals

PITCH_COLORS: dict[str, str] = {
    "FF": "#FF0000",   # Four-seam Fastball - Red
    "SI": "#FF6600",   # Sinker - Orange
    "FC": "#FF9933",   # Cutter - Light Orange
    "SL": "#00BFFF",   # Slider - Blue
    "ST": "#1E90FF",   # Sweeper - Darker Blue
    "SV": "#4169E1",   # Slurve - Royal Blue
    "CU": "#00CC00",   # Curveball - Green
    "KC": "#228B22",   # Knuckle Curve - Forest Green
    "CH": "#9932CC",   # Changeup - Purple
    "FS": "#8B008B",   # Splitter - Dark Purple
    "KN": "#808080",   # Knuckleball - Gray
    "EP": "#A0522D",   # Eephus - Brown
}

# ---------------------------------------------------------------------------
# Canonical pitch-type labels
# ---------------------------------------------------------------------------

PITCH_LABELS: dict[str, str] = {
    "FF": "4-Seam Fastball",
    "SI": "Sinker",
    "FC": "Cutter",
    "SL": "Slider",
    "ST": "Sweeper",
    "SV": "Slurve",
    "CU": "Curveball",
    "KC": "Knuckle Curve",
    "CH": "Changeup",
    "FS": "Splitter",
    "KN": "Knuckleball",
    "EP": "Eephus",
    "IN": "Int. Ball",
    "PO": "Pitchout",
    "CS": "Slow Curve",
    "SC": "Screwball",
    "FA": "Fastball",
    "AB": "Auto Ball",
}

# Fastball-family pitch types (used for emphasis in velocity charts)
FASTBALL_TYPES: set[str] = {"FF", "SI", "FC"}
