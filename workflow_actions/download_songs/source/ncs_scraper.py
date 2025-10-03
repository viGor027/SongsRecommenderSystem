import re
import json
from pathlib import Path
from typing import Optional, List, Dict

import requests
from bs4 import BeautifulSoup

from workflow_actions.paths import DOWNLOAD_DIR, LABELS_DIR, LABELS_PATH


class NCSScraper:
    """
    Object‐oriented scraper for NCS.io music search.

    Downloads songs into `raw/downloaded_songs/`
    and writes out `raw/labels/labels.json` mapping
    "<autor1,autor2>-<tytuł>" to [moods, genres].
    """

    BASE_URL = "https://ncs.io/music-search"

    def __init__(self, pages: Optional[int] = None):
        """
        :param pages: how many search pages to scrape; None = all until empty
        """
        self.pages = pages
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:114.0) "
                    "Gecko/20100101 Firefox/114.0"
                )
            }
        )
        self.labels: Dict[str, List[str]] = {}

        DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        LABELS_DIR.mkdir(parents=True, exist_ok=True)
        for f in DOWNLOAD_DIR.iterdir():
            if f.is_file():
                f.unlink()
        for f in LABELS_DIR.iterdir():
            if f.is_file():
                f.unlink()

    @staticmethod
    def sanitize_title(title: str) -> str:
        """
        Replace any character except alphanumerics, commas and hyphens with underscore.
        This preserves the "autor1,autor2-tytuł" format while cleaning unsafe chars.
        """
        return re.sub(r"[^0-9A-Za-z,\-]+", "_", title).strip("_")

    def fetch_page(self, page: int) -> Optional[BeautifulSoup]:
        """GET a search‐results page and return its BeautifulSoup, or None on error."""
        params = {"q": "", "genre": "", "mood": "", "version": "regular", "page": page}
        try:
            resp = self.session.get(self.BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            print(f"[ERROR] fetch_page({page}): {e}")
            return None

    def parse_songs(self, soup: BeautifulSoup) -> List[Dict]:
        """
        From each <a class="player-play"> element extract:
          - raw_artists: from data-artistraw (comma-separated)
          - track title: from data-track
          - genres: from data-genre
          - moods: from the moods <td>
          - url: from data-url
        Build:
          - display_title: "Artist1, Artist2 - Track"
          - safe_title:     "Artist1,Artist2-Track" (sanitized)
        """
        entries: List[Dict] = []

        for play_btn in soup.select("a.player-play"):
            track = play_btn.get("data-track", "").strip()
            url = play_btn.get("data-url", "").strip()
            raw_genre = play_btn.get("data-genre", "")
            raw_artists = play_btn.get("data-artistraw", "").strip()

            if not track or not url or not raw_artists:
                continue

            genres = [g.strip() for g in raw_genre.split(",") if g.strip()]

            row = play_btn.find_parent("tr")
            mood_cell = row.find("td", attrs={"style": re.compile(r"width\s*:\s*15%")})
            if mood_cell:
                links = mood_cell.find_all("a", href=True)
                moods = (
                    [m.get_text(strip=True) for m in links[1:]]
                    if len(links) > 1
                    else []
                )
            else:
                moods = []

            artists = [a.strip() for a in raw_artists.split(",") if a.strip()]

            display_title = f"{', '.join(artists)} - {track}"

            combined = f"{','.join(artists)}-{track}"

            safe_title = self.sanitize_title(combined)

            entries.append(
                {
                    "title": display_title,
                    "safe_title": safe_title,
                    "genres": genres,
                    "moods": moods,
                    "url": url,
                }
            )

        return entries

    def download_song(self, entry: Dict) -> Optional[str]:
        """
        Download entry['url'] and save to DOWNLOAD_DIR/<safe_title>.<ext>.
        Return filename on success.
        """
        try:
            resp = self.session.get(entry["url"], timeout=20)
            resp.raise_for_status()
            ext = entry["url"].split(".")[-1].split("?")[0]
            fname = f"{entry['safe_title']}.{ext}"
            (DOWNLOAD_DIR / fname).write_bytes(resp.content)
            return fname
        except Exception as e:
            print(f"[ERROR] download '{entry['title']}': {e}")
            return None

    def scrape(self):
        """Loop through pages, download all songs, and write labels.json."""
        page = 1
        while self.pages is None or page <= self.pages:
            print(f"Fetching page {page}...")
            soup = self.fetch_page(page)
            if not soup:
                break

            songs = self.parse_songs(soup)
            if not songs:
                print(f"No more songs on page {page}, stopping.")
                break

            for entry in songs:
                fname = self.download_song(entry)
                if fname:
                    key = Path(fname).stem
                    self.labels[key] = entry["moods"] + entry["genres"]

            print(f"  → Page {page}: {len(songs)} songs processed")
            page += 1

        with open(LABELS_PATH, "w", encoding="utf-8") as f:
            json.dump(self.labels, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(self.labels)} labels to {LABELS_PATH}")


if __name__ == "__main__":
    scraper = NCSScraper(pages=None)
    scraper.scrape()
