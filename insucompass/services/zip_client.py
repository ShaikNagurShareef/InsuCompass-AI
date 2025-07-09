import logging
import requests
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

class ZipCodeData(object):
    """A simple data class to hold the results of our geolocation lookup."""
    def __init__(self, state: str, state_abbr: str, city: str, county: str):
        self.state = state
        self.state_abbr = state_abbr
        self.city = city
        self.county = county

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "state_abbreviation": self.state_abbr,
            "city": self.city,
            "county": self.county
        }

def get_lat_lon_from_zip(zip_code: str) -> Optional[Dict[str, float]]:
    """
    Step 1: Get latitude and longitude from a ZIP code using a simple API.
    We'll use zippopotam.us for this first step.
    """
    url = f"https://api.zippopotam.us/us/{zip_code}"
    logger.info(f"Fetching lat/lon for ZIP code: {zip_code} from {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data.get("places"):
            logger.warning(f"No places found for ZIP code {zip_code}")
            return None
            
        place = data["places"][0]
        return {
            "latitude": float(place["latitude"]),
            "longitude": float(place["longitude"]),
            "state": place["state"],
            "state_abbr": place["state abbreviation"],
            "city": place["place name"]
        }
    except (requests.RequestException, KeyError, ValueError) as e:
        logger.error(f"Failed to get lat/lon for ZIP {zip_code}: {e}")
        return None

def get_county_from_lat_lon(lat: float, lon: float) -> Optional[str]:
    """
    Step 2: Get county information from latitude and longitude using the
    U.S. Census Bureau's Geocoding API.
    """
    url = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
    params = {
        'x': lon,
        'y': lat,
        'benchmark': 'Public_AR_Current',
        'vintage': 'Current_Current',
        'format': 'json'
    }
    logger.info(f"Fetching county for coordinates: (lat={lat}, lon={lon}) from Census Bureau API")
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        geographies = data.get("result", {}).get("geographies", {})
        counties = geographies.get("Counties", [])
        
        if counties:
            county_name = counties[0].get("NAME")
            logger.info(f"Found county: {county_name}")
            return county_name
        else:
            logger.warning(f"No county found for coordinates (lat={lat}, lon={lon})")
            return None
    except (requests.RequestException, KeyError, ValueError) as e:
        logger.error(f"Failed to get county from coordinates: {e}")
        return None

def get_geo_data_from_zip(zip_code: str) -> Optional[ZipCodeData]:
    """
    Orchestrates the two-step process to get state, city, and county from a ZIP code.
    """
    # Step 1: Get Lat/Lon and basic info
    geo_basics = get_lat_lon_from_zip(zip_code)
    if not geo_basics:
        return None
        
    # Step 2: Get County from Lat/Lon
    county = get_county_from_lat_lon(geo_basics["latitude"], geo_basics["longitude"])
    if not county:
        # Fallback: sometimes county info is not available, but we can proceed without it
        logger.warning(f"Could not determine county for ZIP {zip_code}, proceeding without it.")
        county = "Unknown"

    return ZipCodeData(
        state=geo_basics["state"],
        state_abbr=geo_basics["state_abbr"],
        city=geo_basics["city"],
        county=county
    )