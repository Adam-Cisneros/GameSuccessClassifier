import requests
import pandas as pd
import time
import os

API_KEY = "84cd6a1cb1e44e91b11f75ebcad9339e"
BASE_URL = "https://api.rawg.io/api"

PAGE_SIZE = 40
RETRIES = 3
RETRY_DELAY = 5
PAGE_DELAY = 2
MAX_PAGES_PER_RANGE = 42
YEAR_RANGES = [
    ("2010-01-01", "2014-12-31"),
    ("2015-01-01", "2019-12-31"),
    ("2020-01-01", "2025-12-31")
]
OUTPUT_FILE = "rawg_games.csv"

def get_games(page=1, start_date=None, end_date=None, retries=RETRIES):
    """Fetch a page of games from RAWG with retry and date filters."""
    url = f"{BASE_URL}/games"
    params = {
        "key": API_KEY,
        "page": page,
        "page_size": PAGE_SIZE,
        "ordering": "-metacritic"
    }
    if start_date and end_date:
        params["dates"] = f"{start_date},{end_date}"

    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code in [502, 504, 500]:
                print(f"Server error {response.status_code} on page {page}, retry {attempt+1}/{retries}...")
                time.sleep(RETRY_DELAY)
                continue
            response.raise_for_status()
            return response.json().get("results", [])
        except requests.exceptions.RequestException as e:
            print(f"Network error on page {page}: {e} (retry {attempt+1}/{retries})")
            time.sleep(RETRY_DELAY)
    print(f"Failed to fetch page {page} after {retries} attempts.")
    return []


def get_game_details(game_id, retries=RETRIES):
    """Fetch detailed info for one game."""
    url = f"{BASE_URL}/games/{game_id}"
    params = {"key": API_KEY}
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code in [502, 504, 500]:
                print(f"Detail fetch error {response.status_code} for game {game_id}, retry {attempt+1}/{retries}...")
                time.sleep(RETRY_DELAY)
                continue
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Detail request failed for game {game_id}: {e} (retry {attempt+1}/{retries})")
            time.sleep(RETRY_DELAY)
    print(f"Skipping game {game_id} after {retries} failed attempts.")
    return {}


def extract_game_data(detail):
    """Extract relevant fields for CSV."""
    title = detail.get("name")
    genres = ", ".join([g["name"] for g in detail.get("genres", [])]) or None
    # Tags include 'Singleplayer', 'Multiplayer', etc.
    types = [t["name"] for t in detail.get("tags", []) if "player" in t["name"].lower()]
    game_type = ", ".join(types) if types else None
    developer = ", ".join([d["name"] for d in detail.get("developers", [])]) or None
    publisher = ", ".join([p["name"] for p in detail.get("publishers", [])]) or None
    user_rating = detail.get("rating")
    metacritic = detail.get("metacritic")
    return {
        "title": title,
        "genres": genres,
        "type": game_type,
        "developer": developer,
        "publisher": publisher,
        "user_rating": user_rating,
        "metacritic_score": metacritic
    }

def load_existing_data():
    """Load existing CSV if resuming progress."""
    if os.path.exists(OUTPUT_FILE):
        df = pd.read_csv(OUTPUT_FILE)
        print(f"Loaded existing progress: {len(df)} games")
        return df
    return pd.DataFrame(columns=[
        "title", "genres", "type", "developer", "publisher", "user_rating", "metacritic_score"
    ])


def save_checkpoint(df):
    """Save progress to CSV."""
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Checkpoint saved ({len(df)} total games)")

def main():
    all_data = load_existing_data()

    for (start_date, end_date) in YEAR_RANGES:
        print(f"\nFetching games released {start_date} -> {end_date}")
        page = 1
        while page <= MAX_PAGES_PER_RANGE:
            print(f"\nFetching page {page}/{MAX_PAGES_PER_RANGE} ({start_date}–{end_date})")
            games = get_games(page, start_date, end_date)
            if not games:
                print(f"No more games found after page {page} in range {start_date}-{end_date}.")
                break

            for g in games:
                game_id = g.get("id")
                if not game_id:
                    continue
                detail = get_game_details(game_id)
                if not detail:
                    continue
                all_data = pd.concat(
                    [all_data, pd.DataFrame([extract_game_data(detail)])],
                    ignore_index=True
                )
                time.sleep(1)  # delay between detail calls

            # Save progress every page
            save_checkpoint(all_data)
            page += 1
            time.sleep(PAGE_DELAY)

    print(f"\nSaved {len(all_data)} games to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()