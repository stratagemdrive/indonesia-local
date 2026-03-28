"""
fetch_indonesia_news.py

Fetches RSS feeds from Indonesian news sources, translates non-English content,
categorizes stories, and outputs to docs/indonesia_news.json.

Categories: Diplomacy, Military, Energy, Economy, Local Events
Target: 20 stories per category, max age 7 days, Indonesia as primary subject.
"""

import json
import os
import re
import time
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import feedparser
import requests
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COUNTRY = "indonesia"
OUTPUT_DIR = Path("docs")
OUTPUT_FILE = OUTPUT_DIR / f"{COUNTRY}_news.json"

MAX_PER_CATEGORY = 20
MAX_AGE_DAYS = 7
CATEGORIES = ["Diplomacy", "Military", "Energy", "Economy", "Local Events"]

# ---------------------------------------------------------------------------
# RSS Feed Sources
# ---------------------------------------------------------------------------
FEEDS = [
    {
        "name": "The Jakarta Post",
        "url": "https://www.thejakartapost.com/rss/id/news.xml",
        "language": "en",
    },
    {
        "name": "Jakarta Globe",
        "url": "https://jakartaglobe.id/feed",
        "language": "en",
    },
    {
        "name": "Antara News",
        "url": "https://en.antaranews.com/rss/latest-news.xml",
        "language": "en",
    },
    {
        "name": "Tempo English",
        "url": "https://en.tempo.co/rss/id",
        "language": "en",
    },
    # Detik News shut down RSS in Nov 2020 — replaced with Benar News Indonesia
    {
        "name": "Benar News Indonesia",
        "url": "https://www.benarnews.org/rss/indonesian-rss.xml",
        "language": "en",
    },
    {
        "name": "Kompas English",
        "url": "https://go.kompas.com/rss",
        "language": "en",
    },
    {
        "name": "Republika English",
        "url": "https://www.republika.co.id/rss/",
        "language": "id",  # Indonesian — will be translated
    },
    {
        "name": "Indonesia at Melbourne",
        "url": "https://indonesiaatmelbourne.unimelb.edu.au/feed/",
        "language": "en",
    },
    # CNBC Indonesia has no public RSS — replaced with VOA Indonesia
    {
        "name": "VOA Indonesia",
        "url": "https://www.voaindonesia.com/api/zmoqe-eivorg",
        "language": "id",  # Indonesian — will be translated
    },
    {
        "name": "Coconuts Jakarta",
        "url": "https://coconuts.co/jakarta/feed/",
        "language": "en",
    },
]

# ---------------------------------------------------------------------------
# Category keyword mapping (checked against title + summary, lowercased)
# ---------------------------------------------------------------------------
CATEGORY_KEYWORDS = {
    "Diplomacy": [
        "diplomat", "foreign minister", "ambassador", "embassy", "bilateral",
        "multilateral", "treaty", "asean", "united nations", "un ", "summit",
        "foreign policy", "international relations", "sanction", "trade deal",
        "agreement", "cooperation", "g20", "g7", "imf", "world bank",
        "foreign affairs", "ministry of foreign", "state visit", "prabowo visit",
        "diplomatic", "consulate", "protocol",
    ],
    "Military": [
        "military", "army", "navy", "air force", "defense", "defence",
        "weapon", "arms", "soldier", "troops", "war", "conflict", "terror",
        "police", "security force", "tni ", "polri", "patrol", "exercise",
        "drill", "combat", "insurgent", "militant", "separatist", "papua",
        "nat'l police", "national police", "border", "maritime security",
        "coast guard", "commando",
    ],
    "Energy": [
        "energy", "oil", "gas", "coal", "fuel", "electricity", "power plant",
        "solar", "wind power", "renewable", "pertamina", "pln ", "mining",
        "lithium", "nickel", "battery", "carbon", "emission", "climate",
        "geothermal", "hydropower", "lng", "crude", "downstream", "upstream",
        "refinery", "energy transition", "fossil", "green energy",
    ],
    "Economy": [
        "economy", "economic", "gdp", "inflation", "rupiah", "stock",
        "investment", "trade", "export", "import", "tariff", "budget",
        "fiscal", "tax", "revenue", "bank indonesia", "central bank",
        "interest rate", "business", "startup", "industry", "market",
        "manufacturing", "finance", "monetary", "deficit", "debt",
        "agriculture", "tourism", "retail", "consumption",
    ],
    "Local Events": [
        "jakarta", "bali", "surabaya", "bandung", "yogyakarta", "medan",
        "semarang", "makassar", "flood", "earthquake", "volcano", "disaster",
        "local", "province", "district", "municipality", "governor",
        "election", "regional", "community", "cultural", "festival",
        "infrastructure", "transport", "hospital", "school", "village",
        "religious", "mosque", "church", "protest", "demonstration",
    ],
}

# ---------------------------------------------------------------------------
# Indonesia relevance keywords — title or summary must include at least one
# ---------------------------------------------------------------------------
INDONESIA_KEYWORDS = [
    "indonesia", "indonesian", "jakarta", "bali", "java", "sumatra",
    "borneo", "kalimantan", "sulawesi", "papua", "prabowo", "jokowi",
    "joko widodo", "pertamina", "tni", "polri", "rupiah", "komodo",
    "molucca", "maluku",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_date(entry) -> datetime | None:
    """Return a timezone-aware datetime from an RSS entry, or None."""
    for attr in ("published_parsed", "updated_parsed"):
        t = getattr(entry, attr, None)
        if t:
            try:
                return datetime(*t[:6], tzinfo=timezone.utc)
            except Exception:
                pass
    return None


def _clean_html(text: str) -> str:
    """Strip HTML tags from a string."""
    return re.sub(r"<[^>]+>", "", text or "").strip()


def _is_indonesia_relevant(title: str, summary: str) -> bool:
    """Return True if the story appears to be primarily about Indonesia."""
    combined = (title + " " + summary).lower()
    return any(kw in combined for kw in INDONESIA_KEYWORDS)


def _categorize(title: str, summary: str) -> str | None:
    """
    Return the best-matching category, or None if no category fits.
    Checks title first (higher weight), then summary.
    """
    combined = (title + " " + " " + summary).lower()
    scores = {cat: 0 for cat in CATEGORIES}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in combined:
                # Extra weight if keyword is in title
                scores[cat] += 2 if kw in title.lower() else 1
    best_cat = max(scores, key=scores.get)
    return best_cat if scores[best_cat] > 0 else None


def _translate_to_english(text: str, source_lang: str = "auto") -> str:
    """Translate text to English using GoogleTranslator (no API key needed)."""
    if not text:
        return text
    try:
        # Detect language if not provided
        if source_lang == "auto":
            try:
                detected = detect(text)
                if detected == "en":
                    return text
            except LangDetectException:
                pass
        elif source_lang == "en":
            return text

        # Truncate to avoid translator limits (5000 char max per call)
        chunk = text[:4500]
        translated = GoogleTranslator(source="auto", target="en").translate(chunk)
        return translated or text
    except Exception as exc:
        log.warning("Translation failed: %s", exc)
        return text


def _fetch_feed(source: dict) -> list[dict]:
    """Fetch and parse one RSS feed, returning a list of raw story dicts."""
    name = source["name"]
    url = source["url"]
    declared_lang = source.get("language", "en")

    log.info("Fetching: %s — %s", name, url)
    try:
        feed = feedparser.parse(
            url,
            request_headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; IndonesiaNewsScraper/1.0; "
                    "+https://stratagemdrive.github.io/indonesia-local/)"
                )
            },
        )
    except Exception as exc:
        log.error("Failed to fetch %s: %s", name, exc)
        return []

    if feed.bozo and not feed.entries:
        log.warning("Malformed or empty feed: %s", name)
        return []

    stories = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_AGE_DAYS)

    for entry in feed.entries:
        # Date check
        pub_date = _parse_date(entry)
        if pub_date and pub_date < cutoff:
            continue  # Too old

        raw_title = _clean_html(getattr(entry, "title", "") or "")
        raw_summary = _clean_html(getattr(entry, "summary", "") or "")
        link = getattr(entry, "link", "") or ""

        if not raw_title or not link:
            continue

        # Translate if needed
        title = _translate_to_english(raw_title, declared_lang)
        summary = _translate_to_english(raw_summary, declared_lang)
        time.sleep(0.3)  # Be polite to translator service

        # Indonesia relevance filter
        if not _is_indonesia_relevant(title, summary):
            continue

        # Categorize
        category = _categorize(title, summary)
        if category is None:
            category = "Local Events"  # Fallback

        stories.append(
            {
                "title": title,
                "source": name,
                "url": link,
                "published_date": pub_date.isoformat() if pub_date else None,
                "category": category,
            }
        )

    log.info("  → %d relevant stories from %s", len(stories), name)
    return stories


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_existing() -> dict[str, list[dict]]:
    """Load the current JSON file, or return empty category buckets."""
    empty = {cat: [] for cat in CATEGORIES}
    if not OUTPUT_FILE.exists():
        return empty
    try:
        with OUTPUT_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # Validate structure
        if isinstance(data, dict) and all(cat in data for cat in CATEGORIES):
            return data
    except Exception as exc:
        log.warning("Could not load existing JSON (%s). Starting fresh.", exc)
    return empty


def merge_stories(
    existing: dict[str, list[dict]],
    new_stories: list[dict],
) -> dict[str, list[dict]]:
    """
    Merge new stories into existing buckets.
    - Remove stories older than MAX_AGE_DAYS.
    - Deduplicate by URL.
    - Add new stories; if > MAX_PER_CATEGORY, drop oldest entries first.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_AGE_DAYS)

    # Build URL sets and purge stale entries from existing buckets
    result: dict[str, list[dict]] = {}
    existing_urls: set[str] = set()

    for cat in CATEGORIES:
        bucket = existing.get(cat, [])
        fresh = []
        for story in bucket:
            pd = story.get("published_date")
            if pd:
                try:
                    dt = datetime.fromisoformat(pd)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    if dt < cutoff:
                        continue  # Expired
                except Exception:
                    pass
            fresh.append(story)
            existing_urls.add(story["url"])
        result[cat] = fresh

    # Add new stories (deduplicated)
    added_urls: set[str] = set()
    for story in new_stories:
        url = story["url"]
        if url in existing_urls or url in added_urls:
            continue
        cat = story["category"]
        result[cat].append(story)
        added_urls.add(url)

    # Sort each bucket by published_date descending; enforce MAX_PER_CATEGORY
    for cat in CATEGORIES:
        bucket = result[cat]
        bucket.sort(
            key=lambda s: s.get("published_date") or "",
            reverse=True,
        )
        result[cat] = bucket[:MAX_PER_CATEGORY]

    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Fetch all feeds
    all_new: list[dict] = []
    for source in FEEDS:
        stories = _fetch_feed(source)
        all_new.extend(stories)
        time.sleep(1)  # Polite delay between sources

    log.info("Total new candidate stories: %d", len(all_new))

    # 2. Load existing data
    existing = load_existing()

    # 3. Merge
    merged = merge_stories(existing, all_new)

    # 4. Summarise
    for cat in CATEGORIES:
        log.info("  %s: %d stories", cat, len(merged[cat]))

    # 5. Write output
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    log.info("Written → %s", OUTPUT_FILE)


if __name__ == "__main__":
    main()
