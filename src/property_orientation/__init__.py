"""Street-frontage orientation for Australian addresses.

The package answers one question for every address in a study area: *which compass
direction does the property face?* A property is taken to face the street it is addressed
to, so its orientation is the bearing from the address point to the nearest point on that
street.
"""

from property_orientation.geometry import (
    COMPASS_8,
    angular_difference,
    compass8,
    opposite,
    true_azimuth,
)
from property_orientation.orientation import compute_orientation
from property_orientation.streets import gnaf_street_key, normalise_street_name

__version__ = "1.0.0"

__all__ = [
    "COMPASS_8",
    "__version__",
    "angular_difference",
    "compass8",
    "compute_orientation",
    "gnaf_street_key",
    "normalise_street_name",
    "opposite",
    "true_azimuth",
]
