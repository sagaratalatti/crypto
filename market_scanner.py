"""
Market Scanner: Fetches and filters Polymarket markets for trading opportunities.

Connects to:
  - Gamma API (https://gamma-api.polymarket.com) for market metadata
  - CLOB API (via py-clob-client) for order book data and prices
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import config
from reports import log_market_scanned
from price_tracker import record_price_snapshot

logger = logging.getLogger(__name__)

GAMMA_MARKETS_URL = f"{config.GAMMA_API_URL}/markets"
GAMMA_EVENTS_URL = f"{config.GAMMA_API_URL}/events"

# Proper headers to avoid Cloudflare/bot detection blocks
API_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}


def _create_session() -> requests.Session:
    """Create a requests session with retry logic and proper headers."""
    session = requests.Session()
    session.headers.update(API_HEADERS)

    retries = Retry(
        total=3,
        backoff_factor=2,  # 2s, 4s, 8s
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


@dataclass
class MarketInfo:
    """Parsed market data combining Gamma metadata + CLOB pricing."""
    market_id: str
    question: str
    slug: str
    category: str
    end_date: str
    outcomes: list[str]
    outcome_prices: list[float]
    clob_token_ids: list[str]
    volume_24h: float
    volume_total: float
    liquidity: float
    spread: float = 0.0
    best_bid: float = 0.0
    best_ask: float = 0.0
    midpoint: float = 0.0
    order_book_depth: float = 0.0
    tags: list[str] = field(default_factory=list)
    # Time-to-resolution fields (populated by enrichment)
    days_to_resolution: float = -1.0  # -1 = unknown
    theta_regime: str = "unknown"
    theta_size_multiplier: float = 1.0


def fetch_active_markets(limit: int = 100) -> list[dict]:
    """Fetch active, tradeable markets from Gamma API with pagination."""
    all_markets = []
    offset = 0
    session = _create_session()

    while True:
        params = {
            "active": "true",
            "closed": "false",
            "archived": "false",
            "enableOrderBook": "true",
            "limit": limit,
            "offset": offset,
        }

        try:
            resp = session.get(GAMMA_MARKETS_URL, params=params, timeout=30)
            resp.raise_for_status()
            markets = resp.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch markets at offset {offset}: {e}")
            break

        if not markets:
            break

        all_markets.extend(markets)
        offset += limit

        # Safety limit to avoid infinite loops
        if offset > 5000:
            break

        time.sleep(0.3)  # Rate limiting

    logger.info(f"Fetched {len(all_markets)} active markets from Gamma API")
    return all_markets


def test_connectivity() -> dict:
    """Test connectivity to all Polymarket API endpoints. Returns diagnostic info."""
    results = {}
    session = _create_session()

    # Test Gamma API
    try:
        resp = session.get(f"{config.GAMMA_API_URL}/markets",
                          params={"limit": "1"}, timeout=15)
        results["gamma_api"] = {
            "status": resp.status_code,
            "ok": resp.ok,
            "content_type": resp.headers.get("content-type", ""),
            "server": resp.headers.get("server", ""),
            "body_preview": resp.text[:200] if resp.text else "",
        }
    except requests.RequestException as e:
        results["gamma_api"] = {"status": 0, "ok": False, "error": str(e)}

    # Test CLOB API
    try:
        resp = session.get(f"{config.CLOB_API_URL}/", timeout=15)
        results["clob_api"] = {
            "status": resp.status_code,
            "ok": resp.ok,
            "content_type": resp.headers.get("content-type", ""),
            "server": resp.headers.get("server", ""),
            "body_preview": resp.text[:200] if resp.text else "",
        }
    except requests.RequestException as e:
        results["clob_api"] = {"status": 0, "ok": False, "error": str(e)}

    # Test CLOB server time (lightweight endpoint)
    try:
        resp = session.get(f"{config.CLOB_API_URL}/time", timeout=15)
        results["clob_time"] = {
            "status": resp.status_code,
            "ok": resp.ok,
            "body": resp.text[:100] if resp.text else "",
        }
    except requests.RequestException as e:
        results["clob_time"] = {"status": 0, "ok": False, "error": str(e)}

    # General internet check
    try:
        resp = session.get("https://httpbin.org/ip", timeout=10)
        results["internet"] = {
            "ok": True,
            "your_ip": resp.json().get("origin", "unknown") if resp.ok else "unknown",
        }
    except requests.RequestException:
        results["internet"] = {"ok": False, "your_ip": "unreachable"}

    return results


def parse_market(raw: dict) -> Optional[MarketInfo]:
    """Parse raw Gamma API response into a MarketInfo object."""
    try:
        # Parse stringified JSON fields
        outcomes = json.loads(raw.get("outcomes", "[]")) if isinstance(raw.get("outcomes"), str) else raw.get("outcomes", [])
        outcome_prices = json.loads(raw.get("outcomePrices", "[]")) if isinstance(raw.get("outcomePrices"), str) else raw.get("outcomePrices", [])
        clob_token_ids = json.loads(raw.get("clobTokenIds", "[]")) if isinstance(raw.get("clobTokenIds"), str) else raw.get("clobTokenIds", [])

        outcome_prices = [float(p) for p in outcome_prices]

        # Extract tags
        tags = []
        raw_tags = raw.get("tags", [])
        if isinstance(raw_tags, list):
            for t in raw_tags:
                if isinstance(t, dict):
                    tags.append(t.get("label", ""))
                elif isinstance(t, str):
                    tags.append(t)

        return MarketInfo(
            market_id=str(raw.get("id", "")),
            question=raw.get("question", ""),
            slug=raw.get("slug", ""),
            category=raw.get("category", ""),
            end_date=raw.get("endDate", raw.get("end_date_iso", "")),
            outcomes=outcomes,
            outcome_prices=outcome_prices,
            clob_token_ids=clob_token_ids,
            volume_24h=float(raw.get("volume24hr", 0) or 0),
            volume_total=float(raw.get("volume", 0) or 0),
            liquidity=float(raw.get("liquidity", 0) or 0),
            tags=tags,
        )
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.debug(f"Failed to parse market {raw.get('id', '?')}: {e}")
        return None


def enrich_with_orderbook(market: MarketInfo, clob_client) -> MarketInfo:
    """Add order book data (spread, depth, midpoint) from CLOB API."""
    if not market.clob_token_ids:
        return market

    token_id = market.clob_token_ids[0]  # YES token

    try:
        book = clob_client.get_order_book(token_id)

        bids = book.bids if hasattr(book, 'bids') else []
        asks = book.asks if hasattr(book, 'asks') else []

        if bids and asks:
            best_bid = float(bids[0].price) if hasattr(bids[0], 'price') else float(bids[0].get('price', 0))
            best_ask = float(asks[0].price) if hasattr(asks[0], 'price') else float(asks[0].get('price', 0))

            market.best_bid = best_bid
            market.best_ask = best_ask
            market.spread = best_ask - best_bid
            market.midpoint = (best_bid + best_ask) / 2

            # Calculate depth (total USDC within 5% of midpoint)
            depth = 0.0
            for bid in bids:
                price = float(bid.price) if hasattr(bid, 'price') else float(bid.get('price', 0))
                size = float(bid.size) if hasattr(bid, 'size') else float(bid.get('size', 0))
                if price >= market.midpoint * 0.95:
                    depth += price * size
            for ask in asks:
                price = float(ask.price) if hasattr(ask, 'price') else float(ask.get('price', 0))
                size = float(ask.size) if hasattr(ask, 'size') else float(ask.get('size', 0))
                if price <= market.midpoint * 1.05:
                    depth += price * size
            market.order_book_depth = depth

    except Exception as e:
        logger.debug(f"Failed to get order book for {market.question[:50]}: {e}")
        # Fall back to Gamma API prices
        if market.outcome_prices:
            market.midpoint = market.outcome_prices[0]

    return market


def filter_markets(markets: list[MarketInfo]) -> list[MarketInfo]:
    """Filter markets by trading criteria defined in config."""
    filtered = []

    for m in markets:
        # Must have token IDs for trading
        if not m.clob_token_ids or len(m.clob_token_ids) < 2:
            continue

        # Volume filter
        if m.volume_24h < config.MIN_VOLUME_24H:
            continue

        # Liquidity filter
        if m.liquidity < config.MIN_LIQUIDITY:
            continue

        # Price bounds (avoid near-certain or near-impossible outcomes)
        price = m.midpoint if m.midpoint > 0 else (m.outcome_prices[0] if m.outcome_prices else 0.5)
        if price < config.MIN_PRICE or price > config.MAX_PRICE:
            continue

        # Spread filter
        if m.spread > config.MAX_SPREAD and m.spread > 0:
            continue

        filtered.append(m)

    logger.info(f"Filtered to {len(filtered)} tradeable markets from {len(markets)} total")
    return filtered


def scan_markets(clob_client=None) -> list[MarketInfo]:
    """Full pipeline: fetch -> parse -> enrich -> filter -> sort by opportunity."""
    raw_markets = fetch_active_markets()

    parsed = []
    for raw in raw_markets:
        m = parse_market(raw)
        if m:
            parsed.append(m)

    logger.info(f"Parsed {len(parsed)} markets")

    # Enrich with order book data if CLOB client available
    if clob_client:
        enriched = []
        for m in parsed:
            m = enrich_with_orderbook(m, clob_client)
            enriched.append(m)
            time.sleep(0.1)  # Rate limiting
        parsed = enriched

    # Enrich with time-to-resolution (late import to avoid circular dependency)
    from market_intelligence import compute_time_to_resolution
    for m in parsed:
        ttr = compute_time_to_resolution(m)
        if ttr.get("has_end_date"):
            m.days_to_resolution = ttr.get("days_remaining", -1)
            m.theta_regime = ttr.get("theta_regime", "unknown")
            m.theta_size_multiplier = ttr.get("size_multiplier", 1.0)

    # Filter by trading criteria
    tradeable = filter_markets(parsed)

    # Sort by volume (most liquid first)
    tradeable.sort(key=lambda m: m.volume_24h, reverse=True)

    # Log top scanned markets and record price snapshots
    for m in tradeable[:10]:
        price = m.midpoint if m.midpoint > 0 else (m.outcome_prices[0] if m.outcome_prices else 0)
        log_market_scanned(
            market_id=m.market_id,
            question=m.question,
            price=price,
            volume_24h=m.volume_24h,
            liquidity=m.liquidity,
            category=m.category,
        )

    # Record price snapshots for all tradeable markets (feeds backtesting)
    for m in tradeable:
        yes_price = m.outcome_prices[0] if m.outcome_prices else 0
        no_price = m.outcome_prices[1] if len(m.outcome_prices) > 1 else 0
        record_price_snapshot(
            market_id=m.market_id,
            question=m.question,
            yes_price=yes_price,
            no_price=no_price,
            volume_24h=m.volume_24h,
            liquidity=m.liquidity,
            best_bid=m.best_bid,
            best_ask=m.best_ask,
            spread=m.spread,
        )

    return tradeable
