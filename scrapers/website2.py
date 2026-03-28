from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time

# =========================
# SETUP
# =========================
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

url = "https://www.timeshighereducation.com/world-university-rankings/latest/world-ranking"
driver.get(url)

wait = WebDriverWait(driver, 15)

# =========================
# ACCEPT COOKIES
# =========================
try:
    cookie_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(),'Accept')]")))
    cookie_btn.click()
    print("✅ Cookies accepted")
except:
    print("⚠️ No cookie popup")

# =========================
# WAIT FOR TABLE/CARDS
# =========================
wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".css-77a2e3")))

# Scroll to load all cards
print("Scrolling to load all cards...")
for _ in range(10):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(1)

data = []

# =========================
# EXTRACT ALL CARDS
# =========================
print("Extracting all cards...")
cards = driver.find_elements(By.CSS_SELECTOR, ".css-77a2e3")
print(f"Found {len(cards)} cards total")

for idx, card in enumerate(cards):
    try:
        # Extract text from card - adjust selectors based on card structure
        card_text = card.text
        print(f"Card {idx+1}: {card_text[:50]}...")
        
        # TODO: Parse the card_text or find specific elements within card
        # This depends on the card structure
        record = {
            "rank": idx + 1,
            "data": card_text
        }
        data.append(record)
        
    except Exception as e:
        print(f"Error extracting card {idx}: {e}")
        continue

# =========================
# SAVE
# =========================
df = pd.DataFrame(data)
df.to_csv("THE_rankings_selenium.csv", index=False, encoding="utf-8-sig")

print(f"\n🎉 DONE: {len(df)} records saved")

driver.quit()