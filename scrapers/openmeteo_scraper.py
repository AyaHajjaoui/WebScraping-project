# scrapers/openmeteo_scraper.py
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests

import config

logger = logging.getLogger(__name__)


def get_city_coordinates(city_name: str) -> Optional[tuple]:
    """
    Get latitude and longitude for a city using Open-Meteo's geocoding API
    """
    try:
        geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {
            "name": city_name,
            "count": 1,
            "language": "en",
            "format": "json"
        }
        
        response = requests.get(geocoding_url, params=params, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        if data and "results" in data and len(data["results"]) > 0:
            lat = data["results"][0].get("latitude")
            lon = data["results"][0].get("longitude")
            country = data["results"][0].get("country_code", "")
            name = data["results"][0].get("name", city_name)
            return lat, lon, country, name
        else:
            logger.warning(f"No coordinates found for {city_name}")
            return None
    except Exception as e:
        logger.error(f"Error getting coordinates for {city_name}: {e}")
        return None


def get_current_weather(lat: float, lon: float, city_name: str, country: str) -> Optional[Dict]:
    """
    Get current weather for a city using Open-Meteo API
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": [
                "temperature_2m",
                "relative_humidity_2m",
                "apparent_temperature",
                "weather_code",
                "wind_speed_10m"
            ],
            "timezone": "auto"
        }
        
        response = requests.get(url, params=params, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        current = data.get("current", {})
        
        # Convert weather code to description
        weather_code = current.get("weather_code")
        condition = get_weather_description(weather_code)
        
        # Extract data in standard format
        weather_data = {
            "SourceWebsite": "Open-Meteo",
            "City": city_name,
            "Country": country,
            "ScrapeDateTime": datetime.now().isoformat(),
            "Temperature_C": current.get("temperature_2m"),
            "FeelsLike_C": current.get("apparent_temperature"),
            "Humidity_%": current.get("relative_humidity_2m"),
            "WindSpeed_kmh": current.get("wind_speed_10m"),
            "Condition": condition
        }
        
        return weather_data
        
    except Exception as e:
        logger.error(f"Error getting current weather for {city_name}: {e}")
        return None


def get_historical_weather(lat: float, lon: float, city_name: str, country: str, days_back: int) -> List[Dict]:
    """
    Get historical weather data for a city using Open-Meteo API
    """
    historical_data = []
    
    try:
        # Calculate start and end dates
        end_date = datetime.now().date() - timedelta(days=1)
        start_date = end_date - timedelta(days=days_back)
        
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "apparent_temperature",
                "weather_code",
                "wind_speed_10m"
            ],
            "timezone": "auto"
        }
        
        response = requests.get(url, params=params, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        hourly = data.get("hourly", {})
        
        if hourly:
            times = hourly.get("time", [])
            temperatures = hourly.get("temperature_2m", [])
            feels_like = hourly.get("apparent_temperature", [])
            humidity = hourly.get("relative_humidity_2m", [])
            wind_speed = hourly.get("wind_speed_10m", [])
            weather_codes = hourly.get("weather_code", [])
            
            for i in range(len(times)):
                weather_code = weather_codes[i] if i < len(weather_codes) else None
                condition = get_weather_description(weather_code)
                
                weather_data = {
                    "SourceWebsite": "Open-Meteo",
                    "City": city_name,
                    "Country": country,
                    "ScrapeDateTime": datetime.fromisoformat(times[i]).isoformat(),
                    "Temperature_C": temperatures[i] if i < len(temperatures) else None,
                    "FeelsLike_C": feels_like[i] if i < len(feels_like) else None,
                    "Humidity_%": humidity[i] if i < len(humidity) else None,
                    "WindSpeed_kmh": wind_speed[i] if i < len(wind_speed) else None,
                    "Condition": condition
                }
                historical_data.append(weather_data)
            
            logger.info(f"Got {len(historical_data)} historical records for {city_name}")
        
        return historical_data
        
    except Exception as e:
        logger.error(f"Error getting historical weather for {city_name}: {e}")
        return []


def get_weather_description(weather_code: int) -> str:
    """Convert WMO weather code to description"""
    weather_codes = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        56: "Freezing drizzle",
        57: "Dense freezing drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Freezing rain",
        67: "Heavy freezing rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with hail",
        99: "Thunderstorm with heavy hail",
    }
    return weather_codes.get(weather_code, "Unknown")


def scrape_openmeteo(session=None, city_info=None, history_days=0, pass_index=0) -> List[Dict]:
    """
    Main function to scrape Open-Meteo data
    
    Args:
        session: requests session (not used, kept for compatibility)
        city_info: dict with City and Country keys
        history_days: number of days of historical data to fetch
        pass_index: pass number for backfill
    
    Returns:
        List of weather data dictionaries in standard format
    """
    all_data = []
    
    if city_info:
        city_name = city_info.get("City")
        country = city_info.get("Country", "")
        logger.info(f"Scraping Open-Meteo for {city_name}")
        
        # Get coordinates
        coords = get_city_coordinates(city_name)
        if coords:
            lat, lon, detected_country, full_name = coords
            final_country = country if country else detected_country
            final_name = full_name if full_name else city_name
            
            # Get current weather
            current = get_current_weather(lat, lon, final_name, final_country)
            if current:
                all_data.append(current)
                logger.info(f"Got current weather for {final_name}: {current['Temperature_C']}°C")
            else:
                logger.warning(f"Could not get current weather for {final_name}")
            
            # Get historical data if requested
            if history_days > 0:
                logger.info(f"Fetching {history_days} days of historical data for {final_name}")
                historical = get_historical_weather(lat, lon, final_name, final_country, history_days)
                all_data.extend(historical)
                logger.info(f"Got {len(historical)} historical records for {final_name}")
        
        return all_data
    
    else:
        # If no city_info provided, return empty list
        logger.warning("No city_info provided to scrape_openmeteo")
        return []
