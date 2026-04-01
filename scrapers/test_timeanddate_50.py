"""
Quick test: scrape 50 cities from TimeAndDate and verify the
previously-empty columns are now populated.

Run:  python scrapers/test_timeanddate_50.py
"""
import sys
import os
import logging

# Make sure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import requests
from scrapers.timeanddate_scraper import scrape_timeanddate, _ensure_session

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

# ── 50 cities ────────────────────────────────────────────────────────────────
CITIES = [
    {"City": "Beirut",        "Country": "Lebanon",       "TimeAndDate URL": "https://www.timeanddate.com/weather/lebanon/beirut"},
    {"City": "New York",      "Country": "United States", "TimeAndDate URL": "https://www.timeanddate.com/weather/usa/new-york"},
    {"City": "Los Angeles",   "Country": "United States", "TimeAndDate URL": "https://www.timeanddate.com/weather/usa/los-angeles"},
    {"City": "Chicago",       "Country": "United States", "TimeAndDate URL": "https://www.timeanddate.com/weather/usa/chicago"},
    {"City": "Toronto",       "Country": "Canada",        "TimeAndDate URL": "https://www.timeanddate.com/weather/canada/toronto"},
    {"City": "Mexico City",   "Country": "Mexico",        "TimeAndDate URL": "https://www.timeanddate.com/weather/mexico/mexico-city"},
    {"City": "Sao Paulo",     "Country": "Brazil",        "TimeAndDate URL": "https://www.timeanddate.com/weather/brazil/sao-paulo"},
    {"City": "Buenos Aires",  "Country": "Argentina",     "TimeAndDate URL": "https://www.timeanddate.com/weather/argentina/buenos-aires"},
    {"City": "London",        "Country": "United Kingdom","TimeAndDate URL": "https://www.timeanddate.com/weather/uk/london"},
    {"City": "Paris",         "Country": "France",        "TimeAndDate URL": "https://www.timeanddate.com/weather/france/paris"},
    {"City": "Berlin",        "Country": "Germany",       "TimeAndDate URL": "https://www.timeanddate.com/weather/germany/berlin"},
    {"City": "Madrid",        "Country": "Spain",         "TimeAndDate URL": "https://www.timeanddate.com/weather/spain/madrid"},
    {"City": "Rome",          "Country": "Italy",         "TimeAndDate URL": "https://www.timeanddate.com/weather/italy/rome"},
    {"City": "Cairo",         "Country": "Egypt",         "TimeAndDate URL": "https://www.timeanddate.com/weather/egypt/cairo"},
    {"City": "Nairobi",       "Country": "Kenya",         "TimeAndDate URL": "https://www.timeanddate.com/weather/kenya/nairobi"},
    {"City": "Johannesburg",  "Country": "South Africa",  "TimeAndDate URL": "https://www.timeanddate.com/weather/south-africa/johannesburg"},
    {"City": "Dubai",         "Country": "United Arab Emirates", "TimeAndDate URL": "https://www.timeanddate.com/weather/united-arab-emirates/dubai"},
    {"City": "Mumbai",        "Country": "India",         "TimeAndDate URL": "https://www.timeanddate.com/weather/india/mumbai"},
    {"City": "Tokyo",         "Country": "Japan",         "TimeAndDate URL": "https://www.timeanddate.com/weather/japan/tokyo"},
    {"City": "Seoul",         "Country": "South Korea",   "TimeAndDate URL": "https://www.timeanddate.com/weather/south-korea/seoul"},
    {"City": "Singapore",     "Country": "Singapore",     "TimeAndDate URL": "https://www.timeanddate.com/weather/singapore/singapore"},
    {"City": "Sydney",        "Country": "Australia",     "TimeAndDate URL": "https://www.timeanddate.com/weather/australia/sydney"},
    {"City": "Melbourne",     "Country": "Australia",     "TimeAndDate URL": "https://www.timeanddate.com/weather/australia/melbourne"},
    {"City": "Auckland",      "Country": "New Zealand",   "TimeAndDate URL": "https://www.timeanddate.com/weather/new-zealand/auckland"},
    {"City": "Amsterdam",     "Country": "Netherlands",   "TimeAndDate URL": "https://www.timeanddate.com/weather/netherlands/amsterdam"},
    {"City": "Brussels",      "Country": "Belgium",       "TimeAndDate URL": "https://www.timeanddate.com/weather/belgium/brussels"},
    {"City": "Vienna",        "Country": "Austria",       "TimeAndDate URL": "https://www.timeanddate.com/weather/austria/vienna"},
    {"City": "Zurich",        "Country": "Switzerland",   "TimeAndDate URL": "https://www.timeanddate.com/weather/switzerland/zurich"},
    {"City": "Stockholm",     "Country": "Sweden",        "TimeAndDate URL": "https://www.timeanddate.com/weather/sweden/stockholm"},
    {"City": "Oslo",          "Country": "Norway",        "TimeAndDate URL": "https://www.timeanddate.com/weather/norway/oslo"},
    {"City": "Copenhagen",    "Country": "Denmark",       "TimeAndDate URL": "https://www.timeanddate.com/weather/denmark/copenhagen"},
    {"City": "Helsinki",      "Country": "Finland",       "TimeAndDate URL": "https://www.timeanddate.com/weather/finland/helsinki"},
    {"City": "Warsaw",        "Country": "Poland",        "TimeAndDate URL": "https://www.timeanddate.com/weather/poland/warsaw"},
    {"City": "Prague",        "Country": "Czech Republic","TimeAndDate URL": "https://www.timeanddate.com/weather/czech-republic/prague"},
    {"City": "Budapest",      "Country": "Hungary",       "TimeAndDate URL": "https://www.timeanddate.com/weather/hungary/budapest"},
    {"City": "Athens",        "Country": "Greece",        "TimeAndDate URL": "https://www.timeanddate.com/weather/greece/athens"},
    {"City": "Istanbul",      "Country": "Turkey",        "TimeAndDate URL": "https://www.timeanddate.com/weather/turkey/istanbul"},
    {"City": "Moscow",        "Country": "Russia",        "TimeAndDate URL": "https://www.timeanddate.com/weather/russia/moscow"},
    {"City": "Riyadh",        "Country": "Saudi Arabia",  "TimeAndDate URL": "https://www.timeanddate.com/weather/saudi-arabia/riyadh"},
    {"City": "Tehran",        "Country": "Iran",          "TimeAndDate URL": "https://www.timeanddate.com/weather/iran/tehran"},
    {"City": "Karachi",       "Country": "Pakistan",      "TimeAndDate URL": "https://www.timeanddate.com/weather/pakistan/karachi"},
    {"City": "Delhi",         "Country": "India",         "TimeAndDate URL": "https://www.timeanddate.com/weather/india/delhi"},
    {"City": "Dhaka",         "Country": "Bangladesh",    "TimeAndDate URL": "https://www.timeanddate.com/weather/bangladesh/dhaka"},
    {"City": "Colombo",       "Country": "Sri Lanka",     "TimeAndDate URL": "https://www.timeanddate.com/weather/sri-lanka/colombo"},
    {"City": "Bangkok",       "Country": "Thailand",      "TimeAndDate URL": "https://www.timeanddate.com/weather/thailand/bangkok"},
    {"City": "Jakarta",       "Country": "Indonesia",     "TimeAndDate URL": "https://www.timeanddate.com/weather/indonesia/jakarta"},
    {"City": "Manila",        "Country": "Philippines",   "TimeAndDate URL": "https://www.timeanddate.com/weather/philippines/manila"},
    {"City": "Kuala Lumpur",  "Country": "Malaysia",      "TimeAndDate URL": "https://www.timeanddate.com/weather/malaysia/kuala-lumpur"},
    {"City": "Ho Chi Minh City","Country":"Vietnam",      "TimeAndDate URL": "https://www.timeanddate.com/weather/vietnam/ho-chi-minh-city"},
    {"City": "Shanghai",      "Country": "China",         "TimeAndDate URL": "https://www.timeanddate.com/weather/china/shanghai"},
]

TARGET_COLS = ["FeelsLike_C", "WindSpeed_kmh", "Condition", "Precipitation"]

def main():
    session = _ensure_session()
    all_rows = []
    total = len(CITIES)

    print(f"\n{'='*60}")
    print(f"  TimeAndDate Scraper Test — {total} cities")
    print(f"{'='*60}\n")

    for i, city in enumerate(CITIES, 1):
        print(f"[{i:02d}/{total}] Scraping {city['City']}...", end=" ", flush=True)
        try:
            rows = scrape_timeanddate(session, city, history_days=1, pass_index=0)
            # Only keep the current-weather row (has a full ISO timestamp)
            current = [r for r in rows if "T" in str(r.get("ScrapeDateTime", ""))]
            if current:
                r = current[0]
                filled = sum(1 for c in TARGET_COLS if r.get(c) is not None)
                print(f"OK  {r['Temperature_C']}C  FL={r['FeelsLike_C']}  "
                      f"Wind={r['WindSpeed_kmh']}  Cond={repr(r['Condition'])}  "
                      f"Precip={r['Precipitation']}  [{filled}/{len(TARGET_COLS)} target cols filled]")
                all_rows.extend(rows)
            else:
                print("WARN  No current-weather row returned")
        except Exception as e:
            print(f"FAIL  ERROR: {e}")

    if not all_rows:
        print("\nNo data collected — check network / selectors.")
        return

    df = pd.DataFrame(all_rows)
    current_df = df[df["ScrapeDateTime"].str.contains("T", na=False)]

    print(f"\n{'='*60}")
    print(f"  Fill-rate for current-weather rows ({len(current_df)} cities scraped)")
    print(f"{'='*60}")
    for col in TARGET_COLS:
        filled   = current_df[col].notna().sum()
        pct      = filled / len(current_df) * 100 if len(current_df) else 0
        status   = "OK  " if pct >= 80 else ("WARN" if pct >= 40 else "FAIL")
        print(f"  {status}  {col:<20} {filled:>3}/{len(current_df)}  ({pct:.0f}%)")

    print(f"\n  Total rows collected: {len(df)}")
    print("  Done.\n")

if __name__ == "__main__":
    main()
