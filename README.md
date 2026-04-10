# Polymarket Trading Strategy Bot

A multi-signal trading bot for [Polymarket](https://polymarket.com) prediction markets, optimized for small bankrolls ($99 USDC).

## Strategy Overview

The bot uses 4 complementary strategies to find edge in prediction markets:

| Strategy | How It Works | Edge Source | Risk |
|----------|-------------|-------------|------|
| **Spread Capture** | Place limit orders inside bid-ask spread | Earn the spread gap | Low |
| **Value Betting** | Detect YES+NO mispricing (sum != 1.0) | Arbitrage-like edge | Low-Med |
| **Mean Reversion** | Fade extreme short-term price moves | Price overshoot | Medium |
| **Momentum** | Ride volume surges on breaking news | Information flow | Med-High |

### Risk Management
- **Quarter-Kelly** position sizing (conservative for small bankrolls)
- Max 25% of bankroll per position
- Max 5 concurrent positions
- 20% bankroll reserve (never fully deployed)
- Stop-loss and take-profit on every position
- Minimum $5 position size (avoid dust)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your wallet details

# 3. Scan markets (no wallet needed)
python main.py scan

# 4. Run full analysis
python main.py analyze

# 5. Paper trade (DRY_RUN=true by default)
python main.py trade

# 6. Continuous trading loop
python main.py run
```

## Commands

| Command | Description |
|---------|-------------|
| `python main.py scan` | Scan markets and show tradeable opportunities |
| `python main.py analyze` | Full analysis with signal generation and position sizing |
| `python main.py trade` | Execute one round of trades |
| `python main.py run` | Continuous trading loop (Ctrl+C to stop) |
| `python main.py portfolio` | Show current portfolio, positions, and P&L |
| `python main.py cancel-all` | Cancel all open orders |

## Configuration

All settings are in `.env` (copy from `.env.example`):

```
PRIVATE_KEY=           # Ethereum private key (Polygon network)
WALLET_ADDRESS=        # Your wallet address
BANKROLL=99.0          # Starting capital in USDC
MAX_POSITION_PCT=0.25  # Max 25% per position
MIN_EDGE=0.05          # Minimum 5% edge required
KELLY_FRACTION=0.25    # Quarter-Kelly sizing
DRY_RUN=true           # Paper trading mode
```

## Architecture

```
main.py            - CLI runner and trading loop
market_scanner.py  - Fetches and filters markets from Gamma + CLOB APIs
strategy.py        - Signal generation (4 strategies)
risk_manager.py    - Kelly Criterion sizing + portfolio constraints
executor.py        - Order placement via py-clob-client
portfolio.py       - Position tracking and P&L persistence
config.py          - Environment-based configuration
```

## Requirements

- Python 3.9+
- USDC on Polygon network (for live trading)
- A wallet private key (MetaMask, etc.)

## Disclaimer

This is experimental trading software. Prediction markets carry risk of total loss. Never trade more than you can afford to lose. Past performance does not guarantee future results. Start with DRY_RUN=true.
