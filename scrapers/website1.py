from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import pandas as pd
import time
import os

DEBUG = False
PROGRESS_FILE = "scraping_progress.txt"
OUTPUT_CSV = "qs_world_rankings.csv"


def get_last_scraped_page():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return int(f.read().strip())
        except:
            return 1
    return 1

def save_progress(page_number):
    with open(PROGRESS_FILE, 'w') as f:
        f.write(str(page_number))

def load_existing_data():
    if os.path.exists(OUTPUT_CSV):
        try:
            df = pd.read_csv(OUTPUT_CSV)
            print(f"Loaded {len(df)} existing records from {OUTPUT_CSV}")
            return df, set(df['University Name'].dropna())
        except:
            print(f"Could not load existing CSV, starting fresh.")
            return pd.DataFrame(), set()
    return pd.DataFrame(), set()

def save_data(df):
    df = df.drop_duplicates(subset=["University Name"])
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Saved {len(df)} records to {OUTPUT_CSV}")

def dismiss_consent(driver):
    selectors = [
        (By.ID, "onetrust-accept-btn-handler"),
        (By.CSS_SELECTOR, "button.cookie-accept"),
        (By.XPATH, "//button[contains(text(),'Accept')]"),
    ]
    for by, sel in selectors:
        try:
            btn = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((by, sel)))
            btn.click()
            print("Dismissed consent popup.")
            time.sleep(1)
            return
        except TimeoutException:
            continue

def find_rows(driver):
    candidates = [
        "div._qs_ranking_table div[class*='row']:not([class*='header'])",
        "div[class*='ranking'] div[class*='ind']",
        "tr.ranking-institution-row",
        "div[data-rank]",
        "[class*='uni_row']",
        "[class*='uniRow']",
    ]
    for sel in candidates:
        rows = driver.find_elements(By.CSS_SELECTOR, sel)
        if len(rows) > 1:
            if DEBUG:
                print(f"[DEBUG] Matched selector: {sel!r}  ({len(rows)} rows)")
            return rows
    return []

def extract_text(el, *css_selectors):
    for sel in css_selectors:
        try:
            text = el.find_element(By.CSS_SELECTOR, sel).text.strip()
            if text:
                return text
        except NoSuchElementException:
            continue
    return None

def scrape_cards(driver, seen_names):
    rows = find_rows(driver)
    universities = []
    for row in rows:
        name = extract_text(row,
            "[class*='uni-name']",
            "[class*='uniName']",
            "[class*='name']",
            "a[href*='/universities/']",
            "h3", "h2", "strong",
        )
        if not name:
            try:
                a = row.find_element(By.CSS_SELECTOR, "a[href*='/universities/']")
                name = a.text.strip()
            except NoSuchElementException:
                continue
        if not name or name in seen_names:
            continue
        seen_names.add(name)

        rank = extract_text(row, "[class*='rank']:not([class*='ranking'])", "span[class*='rank']", "td:first-child")
        location = extract_text(row, "[class*='location']", "[class*='country']")
        score = extract_text(row, "[class*='overall']", "[class*='score']")

        link = None
        try:
            link = row.find_element(By.CSS_SELECTOR, "a[href*='/universities/']").get_attribute("href")
        except NoSuchElementException:
            pass

        universities.append({
            "Rank": rank,
            "University Name": name,
            "Location": location,
            "Score": score,
            "Profile Link": link,
        })
    return universities

# pagination
def navigate_to_page(driver, page_number, base_url="https://www.topuniversities.com/world-university-rankings"):
    """Navigate to a specific page using URL parameters."""
    url = f"{base_url}?items_per_page=150&page={page_number}"
    try:
        driver.get(url)
        time.sleep(3) 
        return True
    except Exception as e:
        print(f"Error navigating to page {page_number}: {e}")
        return False



def scrape_qs_numbered_pages():
    url = "https://www.topuniversities.com/world-university-rankings"

    existing_df, seen_names = load_existing_data()
    start_page = get_last_scraped_page() + 1
    
    print(f"Starting from page {start_page}")
    print(f"Already have {len(existing_df)} universities")

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(options=options)
    
    try:
        driver.get(url)
        time.sleep(6)
        dismiss_consent(driver)
        time.sleep(3)

        universities = []

        if start_page == 1:
            # print for trials and debugging
            print("Scraping page 1...") 
            universities.extend(scrape_cards(driver, seen_names))
            save_progress(1)
            start_page = 2
  
        total_pages = 11

        for page in range(start_page, total_pages + 1):
            try:
                print(f"Scraping page {page}...")
                if navigate_to_page(driver, page):
                    deadline = time.time() + 20
                    loaded = False
                    while time.time() < deadline:
                        rows = find_rows(driver)
                        if len(rows) > 0:
                            loaded = True
                            break
                        time.sleep(0.5)
                    
                    if not loaded:
                        print(f"Warning: No rows found on page {page}, retrying...")
                        time.sleep(2)
                        if not navigate_to_page(driver, page):
                            print(f"Failed to load page {page}, skipping...")
                            continue
                    
                    page_universities = scrape_cards(driver, seen_names)
                    universities.extend(page_universities)
                    print(f"  Found {len(page_universities)} new universities")
                    
                    save_progress(page)
                    
                    if page % 5 == 0 or page == total_pages:
                        combined_df = pd.concat([existing_df, pd.DataFrame(universities)], ignore_index=True)
                        save_data(combined_df)
                else:
                    print(f"Could not navigate to page {page}, skipping...")
                    continue
                        
            except Exception as e:
                print(f"Error on page {page}: {e}")
                print(f"Saving progress and will resume on next run...")
                break


        if universities:
            combined_df = pd.concat([existing_df, pd.DataFrame(universities)], ignore_index=True)
            save_data(combined_df)
            print(f"\nTotal universities scraped: {len(combined_df)}")
        else:
            print("\nNo new universities scraped.")

    except Exception as e:
        print(f"Fatal error: {e}")
        print(f"Saving current progress...")
        if universities:
            combined_df = pd.concat([existing_df, pd.DataFrame(universities)], ignore_index=True)
            save_data(combined_df)
    
    finally:
        driver.quit()



if __name__ == "__main__":
    scrape_qs_numbered_pages()
    
    # print a summary
    if os.path.exists(OUTPUT_CSV):
        df = pd.read_csv(OUTPUT_CSV)
        print(f"\n{'='*50}")
        print(f"Final Summary:")
        print(f"{'='*50}")
        print(f"Total universities in database: {len(df)}")
        print(f"Columns: {', '.join(df.columns)}")
        print(f"\nFirst 5 records:")
        print(df.head())
        print(f"\nFile saved: {OUTPUT_CSV}")