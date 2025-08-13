import requests
import json
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os

# --- CONFIGURATION ---
MASTER_DB_FILENAME = "master_food_database.json"
DINING_HALLS = ["Earhart", "Ford", "Hillenbrand", "Wiley", "Windsor", "1Bowl", "Pete's Za", 
                "The Gathering Place featuring Sushi Boss"] #this will have more places when Purdue opens
MAX_THREADS = 20 


# Food can change daily, so we will scrape a range of dates
START_DATE = date.today() - timedelta(days=1)
END_DATE = date.today() + timedelta(days=11) 

# --- UTILITY FUNCTIONS (Unchanged) ---
def load_master_database():
    if os.path.exists(MASTER_DB_FILENAME):
        with open(MASTER_DB_FILENAME, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def fetch_menu_worker(hall, day_str):
    """Worker to fetch one menu for one hall on one day."""
    url = f"https://api.hfs.purdue.edu/menus/v2/locations/{hall}/{day_str}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException:
        pass
    return None

def fetch_nutrition_worker(item_id):
    """Worker to fetch nutrition for one item."""
    try:
        url = f"https://api.hfs.purdue.edu/menus/v2/items/{item_id}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return item_id, resp.json()
    except Exception:
        return item_id, None

# --- MAIN EXECUTION ---
print("--- Purdue Food Database BULK SCRAPER ---")

# 1. Load existing database (if any)
master_items = load_master_database()
print(f"📖 Loaded {len(master_items)} items from existing database.")

# 2. Generate all dates and all jobs
dates_to_scrape = []
current_date = START_DATE
while current_date <= END_DATE:
    dates_to_scrape.append(current_date.strftime("%Y-%m-%d"))
    current_date += timedelta(days=1)

menu_jobs = [(hall, day) for hall in DINING_HALLS for day in dates_to_scrape]
print(f"\n🎯 Preparing to scrape {len(menu_jobs)} menus from {START_DATE} to {END_DATE}...")

# 3. Fetch all menus in parallel
items_to_fetch_nutrition = set()

with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    futures = [executor.submit(fetch_menu_worker, job[0], job[1]) for job in menu_jobs]
    for future in tqdm(as_completed(futures), total=len(menu_jobs), desc="📅 Fetching menus"):
        menu_data = future.result()
        if not menu_data:
            continue
        
        # Find unique items we haven't seen before
        for meal in menu_data.get("Meals", []):
            for station in meal.get("Stations", []):
                for item in station.get("Items", []):
                    item_id = item.get("ID")
                    if item_id and item_id not in master_items:
                        items_to_fetch_nutrition.add(item_id)

new_item_count = len(items_to_fetch_nutrition)
if not new_item_count:
    print("No new food items discovered in this date range.")
    exit()

print(f"Discovered {new_item_count} new unique items across the entire date range. Fetching details...")


# 4. Fetch nutrition for ONLY the new, unique items
with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    futures = [executor.submit(fetch_nutrition_worker, item_id) for item_id in items_to_fetch_nutrition]
    
    for future in tqdm(as_completed(futures), total=len(futures), desc=" Fetching nutrition"):
        item_id, item_data = future.result()
        if item_data:
            master_items[item_id] = item_data # Add it to our master collection

# 5. Save the final, massive database
print(f"Saving updated database...")
with open(MASTER_DB_FILENAME, 'w', encoding='utf-8') as f:
    json.dump(master_items, f, indent=2)

print(f"Success! Master database now contains {len(master_items)} unique items.")
print("This bulk scrape is complete. You are all set.")

