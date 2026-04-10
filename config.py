"""
Configuration for Polymarket Trading Strategy Bot.
Loads settings from environment variables with sensible defaults.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── API & Chain ─────────────────────────────────────────────────────────────
CLOB_API_URL = "https://clob.polymarket.com"
GAMMA_API_URL = "https://gamma-api.polymarket.com"
CHAIN_ID = 137  # Polygon

# ─── Wallet ──────────────────────────────────────────────────────────────────
PRIVATE_KEY = os.getenv("PRIVATE_KEY", "")
WALLET_ADDRESS = os.getenv("WALLET_ADDRESS", "")
SIGNATURE_TYPE = int(os.getenv("SIGNATURE_TYPE", "0"))
FUNDER_ADDRESS = os.getenv("FUNDER_ADDRESS", "")

# ─── Trading Parameters ─────────────────────────────────────────────────────
BANKROLL = float(os.getenv("BANKROLL", "99.0"))

# Maximum % of bankroll per single position (risk control)
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.25"))

# Minimum edge (estimated_prob - market_price) required to enter a trade
MIN_EDGE = float(os.getenv("MIN_EDGE", "0.05"))

# Kelly Criterion fraction (0.25 = quarter Kelly, conservative for small bankrolls)
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.25"))

# Max number of simultaneous open positions
MAX_CONCURRENT_POSITIONS = int(os.getenv("MAX_CONCURRENT_POSITIONS", "5"))

# Dry run mode - simulates trades without executing
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

# ─── Market Scanning ────────────────────────────────────────────────────────
# Minimum 24h volume (USDC) to consider a market
MIN_VOLUME_24H = float(os.getenv("MIN_VOLUME_24H", "5000"))

# Minimum liquidity depth (USDC) in the order book
MIN_LIQUIDITY = float(os.getenv("MIN_LIQUIDITY", "1000"))

# Maximum bid-ask spread to consider (wide spreads = hard to fill)
MAX_SPREAD = float(os.getenv("MAX_SPREAD", "0.10"))

# Price bounds - avoid markets already near 0 or 1 (resolved or near-certain)
MIN_PRICE = float(os.getenv("MIN_PRICE", "0.05"))
MAX_PRICE = float(os.getenv("MAX_PRICE", "0.95"))

# ─── Strategy Intervals ─────────────────────────────────────────────────────
SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "300"))  # 5 min
ORDER_CHECK_INTERVAL = int(os.getenv("ORDER_CHECK_INTERVAL", "60"))  # 1 min

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
