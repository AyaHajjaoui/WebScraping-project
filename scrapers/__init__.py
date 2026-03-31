# scrapers/__init__.py
from .openmeteo_scraper import scrape_openmeteo
from .timeanddate_scraper import scrape_timeanddate
from .wunderground_scraper import scrape_wunderground

__all__ = [
    'scrape_openmeteo',
    'scrape_timeanddate', 
    'scrape_wunderground'
]