import logging
logger = logging.getLogger(__name__)
from app.utils.config import AppConfig
import yfinance as yf
import feedparser
import os
import json
from datetime import datetime, timedelta, timezone

class NewsDataManager:
    """
    Owns all news fetching, caching and deduplication.
    Mirrors FinanceDataManager's responsibility but for text data.
    """
    
    def __init__(self, cache_dir: str, config: AppConfig):
        self.cache_dir = cache_dir
        self._config = config
        self._cache: dict[str, list[dict]] = {}  # in-memory: ticker -> headlines
        os.makedirs(self.cache_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public interface                                                   #
    # ------------------------------------------------------------------ #

    def get_headlines(self, ticker: str, 
                      force: bool = False) -> list[dict]:
        """
        Main entry point. Returns deduplicated, age-filtered headlines.
        Caches to disk so headlines persist between app restarts.
        """
        if not force and ticker in self._cache:
            return self._cache[ticker]
        
        # Load existing disk cache
        existing = self._load_from_disk(ticker)
        
        # Network fetch
        fresh = []
        fresh.extend(self._fetch_yfinance(ticker))
        fresh.extend(self._fetch_rss(ticker))
        # fresh.extend(self._fetch_finviz(ticker))  # add later
        # fresh.extend(self._fetch_alpha_vantage(ticker))  #TODO add later
        
        # Merge: existing + fresh, then filter and dedup
        merged = existing + fresh
        merged = self._filter_by_age(merged) # drop headlines older than max_age
        merged = self._deduplicate(merged) # remove duplicates across old+new
        merged.sort(key=lambda h: h.get('published') or datetime.min.replace(tzinfo=timezone.utc), 
                    reverse=True) # most recent first
        self._cache[ticker] = merged
        self._persist(ticker, merged)
        
        new_count = len(merged) - len(existing)
        logger.info(f"[NEWS] {ticker}: {len(merged)} total headlines "
                    f"({new_count} new) from "
                    f"{len(set(h['source'] for h in fresh))} sources")
        return merged

    # ------------------------------------------------------------------ #
    # Private helpers                                                    #
    # ------------------------------------------------------------------ #

    def _load_from_disk(self, ticker: str) -> list[dict]:
        """Load persisted headlines, filtering stale ones by age."""
        path = os.path.join(self.cache_dir, f"{ticker}_news.json")
        if not os.path.exists(path):
            return []
        try:
            with open(path) as f:
                headlines = json.load(f)
            # Deserialise datetime strings back to datetime objects
            for h in headlines:
                if h.get('published'):
                    h['published'] = datetime.fromisoformat(h['published'])
            # Apply age filter to disk cache too
            return self._filter_by_age(headlines)
        except (json.JSONDecodeError, IOError, ValueError) as e:
            logger.warning(f"[NEWS] Disk cache load failed for {ticker}: {e}")
            return []

    def _fetch_yfinance(self, ticker: str) -> list[dict]:
        try:
            items = yf.Ticker(ticker).news or []
            return [{'title':     item['title'],
                     'published': datetime.fromtimestamp(
                         item.get('providerPublishTime', 0), 
                         tz=timezone.utc),
                     'source':    'yfinance',
                     'url':       item.get('link', '')}
                    for item in items if 'title' in item]
        except Exception as e:
            logger.warning(f"[NEWS] yfinance failed for {ticker}: {e}")
            return []

    def _fetch_rss(self, ticker: str) -> list[dict]:
        try:
            url = (f"https://feeds.finance.yahoo.com/rss/2.0/headline"
                   f"?s={ticker}&region=US&lang=en-US")
            feed = feedparser.parse(url)
            results = []
            for entry in feed.entries:
                if not hasattr(entry, 'title'):
                    continue
                published = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    published = datetime(*entry.published_parsed[:6],
                                        tzinfo=timezone.utc)
                results.append({'title':     entry.title,
                                'published': published,
                                'source':    'rss',
                                'url':       getattr(entry, 'link', '')})
            return results
        except Exception as e:
            logger.warning(f"[NEWS] RSS failed for {ticker}: {e}")
            return []

    def _filter_by_age(self, headlines: list[dict]) -> list[dict]:
        max_age = self._config.get("news_max_age_days")
        print("max_age:", max_age)
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age)
        return [h for h in headlines 
                if h.get('published') is None or h['published'] >= cutoff]

    def _deduplicate(self, headlines: list[dict], 
                     threshold: float = 0.85) -> list[dict]:
        seen = []
        for h in headlines:
            tokens = set(h['title'].lower().split())
            is_dup = any(
                len(tokens & set(s['title'].lower().split())) /
                max(len(tokens | set(s['title'].lower().split())), 1) > threshold
                for s in seen
            )
            if not is_dup:
                seen.append(h)
        return seen

    def _persist(self, ticker: str, headlines: list[dict]) -> None:
        """Save to disk so headlines survive app restarts."""
        path = os.path.join(self.cache_dir, f"{ticker}_news.json")
        serialisable = [
            {**h, 'published': h['published'].isoformat() 
             if h.get('published') else None}
            for h in headlines
        ]
        with open(path, 'w') as f:
            json.dump(serialisable, f, indent=2)