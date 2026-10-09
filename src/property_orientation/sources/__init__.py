"""Downloaders for the open datasets used by the pipeline."""

# The Overpass API rejects (HTTP 406) any User-Agent containing the word "property", which
# real-estate scrapers use, so the HTTP identity uses the method's name instead.
USER_AGENT = "frontage-orientation/1.0 (open-source GIS research; github.com/midhunshyam)"
