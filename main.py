#!/usr/bin/env python3
"""
Polymarket Trading Strategy Bot

A multi-signal trading bot for Polymarket prediction markets,
optimized for small bankrolls ($99 USDC).

Usage:
    python main.py scan          # Scan markets and show opportunities
    python main.py analyze       # Full analysis with position sizing
    python main.py trade         # Execute trades (requires wallet config)
    python main.py portfolio     # Show portfolio status
    python main.py report        # Full P&L report with timestamps
    python main.py run           # Continuous trading loop
    python main.py cancel-all    # Cancel all open orders

Environment:
    Set DRY_RUN=true in .env for paper trading (default).
    Set DRY_RUN=false and configure wallet keys for live trading.
"""

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timezone

from tabulate import tabulate

import config
from market_scanner import scan_markets, MarketInfo
from strategy import generate_signals, TradeOpportunity
from risk_manager import size_positions, validate_trade, calculate_stop_loss, calculate_take_profit
from executor import (
    create_clob_client,
    execute_limit_order,
    cancel_all_orders,
)
from portfolio import (
    load_state,
    save_state,
    get_portfolio_state,
    add_position,
    update_position_prices,
    check_stop_loss_take_profit,
    get_portfolio_summary,
)
from reports import (
    generate_performance_report,
    export_trades_csv,
    export_report_json,
    export_equity_curve_csv,
)

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("polymarket_bot")

# Graceful shutdown
_running = True


def _signal_handler(sig, frame):
    global _running
    logger.info("Shutdown signal received, finishing current cycle...")
    _running = False


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ─── Commands ────────────────────────────────────────────────────────────────

def cmd_scan(args):
    """Scan markets and display tradeable opportunities."""
    logger.info("Scanning Polymarket for trading opportunities...")

    clob_client = None
    try:
        clob_client = create_clob_client()
    except Exception:
        logger.warning("Running without CLOB client (read-only, no order book data)")

    markets = scan_markets(clob_client)

    if not markets:
        print("\nNo tradeable markets found matching criteria.")
        print(f"  Min volume:    ${config.MIN_VOLUME_24H:,.0f}")
        print(f"  Min liquidity: ${config.MIN_LIQUIDITY:,.0f}")
        print(f"  Max spread:    {config.MAX_SPREAD:.0%}")
        return

    # Display results
    table_data = []
    for m in markets[:20]:
        price = m.midpoint if m.midpoint > 0 else (m.outcome_prices[0] if m.outcome_prices else 0)
        table_data.append([
            m.question[:55] + ("..." if len(m.question) > 55 else ""),
            f"${m.volume_24h:>10,.0f}",
            f"${m.liquidity:>8,.0f}",
            f"{price:.3f}",
            f"{m.spread:.3f}" if m.spread > 0 else "N/A",
            m.category or "—",
        ])

    print(f"\n{'='*100}")
    print(f"  POLYMARKET SCANNER — {len(markets)} tradeable markets found")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*100}\n")

    headers = ["Market", "Vol 24h", "Liquidity", "Price", "Spread", "Category"]
    print(tabulate(table_data, headers=headers, tablefmt="simple"))
    print(f"\nShowing top {min(20, len(markets))} of {len(markets)} markets.")


def cmd_analyze(args):
    """Full analysis: scan + signal generation + position sizing."""
    logger.info("Running full market analysis...")

    clob_client = None
    try:
        clob_client = create_clob_client()
    except Exception:
        logger.warning("Running without CLOB client")

    # Scan markets
    markets = scan_markets(clob_client)
    if not markets:
        print("No tradeable markets found.")
        return

    # Generate signals
    opportunities = generate_signals(markets)
    if not opportunities:
        print(f"\nNo opportunities with edge >= {config.MIN_EDGE:.0%} found.")
        print(f"Scanned {len(markets)} markets.")
        return

    # Size positions
    state = load_state()
    portfolio = get_portfolio_state(state)
    sized = size_positions(opportunities, portfolio)

    # Display analysis
    print(f"\n{'='*100}")
    print(f"  TRADING ANALYSIS — Bankroll: ${portfolio.bankroll:.2f} | "
          f"Available: ${portfolio.available_capital:.2f} | "
          f"Positions: {portfolio.num_positions}/{config.MAX_CONCURRENT_POSITIONS}")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*100}\n")

    print(f"  Markets scanned:  {len(markets)}")
    print(f"  Signals found:    {len(opportunities)}")
    print(f"  Sized positions:  {len(sized)}")
    print()

    if sized:
        print("  RECOMMENDED TRADES:")
        print("  " + "-" * 96)

        table_data = []
        for opp in sized:
            table_data.append([
                opp.signal.value,
                opp.side.value,
                opp.market.question[:40] + "...",
                f"{opp.entry_price:.3f}",
                f"{opp.edge:.3f}",
                f"{opp.confidence:.2f}",
                f"{opp.kelly_size:.1%}",
                f"${opp.dollar_size:.2f}",
            ])

        headers = ["Signal", "Side", "Market", "Price", "Edge", "Conf", "Kelly%", "Size$"]
        print(tabulate(table_data, headers=headers, tablefmt="simple"))

        total = sum(o.dollar_size for o in sized)
        print(f"\n  Total deployment: ${total:.2f} / ${portfolio.available_capital:.2f} available")
        print()

    # Show all raw signals too
    if opportunities and len(opportunities) > len(sized):
        print(f"\n  ALL SIGNALS ({len(opportunities)} total, sorted by expected value):")
        print("  " + "-" * 96)
        for i, opp in enumerate(opportunities[:15]):
            ev = opp.edge * opp.confidence
            print(
                f"  {i+1:2d}. [{opp.signal.value:16s}] "
                f"{opp.side.value:8s} | "
                f"Edge={opp.edge:.3f} | "
                f"Conf={opp.confidence:.2f} | "
                f"EV={ev:.4f} | "
                f"{opp.market.question[:50]}"
            )


def cmd_trade(args):
    """Execute trades based on current analysis."""
    logger.info("Executing trading cycle...")

    mode = "DRY RUN" if config.DRY_RUN else "LIVE"
    print(f"\n  Mode: {mode}")

    if not config.DRY_RUN and not config.PRIVATE_KEY:
        print("ERROR: PRIVATE_KEY required for live trading. Set in .env file.")
        sys.exit(1)

    clob_client = create_clob_client()

    # Scan and analyze
    markets = scan_markets(clob_client)
    opportunities = generate_signals(markets)

    # Load portfolio state
    state = load_state()
    portfolio = get_portfolio_state(state)

    # Size positions
    sized = size_positions(opportunities, portfolio)

    if not sized:
        print("No qualifying trades this cycle.")
        return

    print(f"\n  Executing {len(sized)} trades...\n")

    for opp in sized:
        # Final validation
        ok, reason = validate_trade(opp, portfolio)
        if not ok:
            logger.info(f"Trade rejected: {reason}")
            continue

        # Calculate exit levels
        stop_loss = calculate_stop_loss(opp)
        take_profit = calculate_take_profit(opp)

        # Execute
        order = execute_limit_order(clob_client, opp)

        if order:
            state = add_position(state, order, stop_loss, take_profit)
            portfolio = get_portfolio_state(state)

            print(
                f"  {'[SIM]' if config.DRY_RUN else '[LIVE]'} "
                f"{opp.signal.value:16s} | "
                f"{opp.side.value:8s} | "
                f"${opp.dollar_size:7.2f} @ {opp.entry_price:.3f} | "
                f"SL={stop_loss:.3f} TP={take_profit:.3f} | "
                f"{opp.market.question[:40]}"
            )

    print(f"\n{get_portfolio_summary(state)}")


def cmd_portfolio(args):
    """Display current portfolio status."""
    state = load_state()

    clob_client = None
    try:
        clob_client = create_clob_client()
        state = update_position_prices(state, clob_client)
    except Exception:
        pass

    print(get_portfolio_summary(state))

    # Show recent closed positions
    closed = state.get("closed_positions", [])
    if closed:
        print("  RECENT CLOSED POSITIONS:")
        print("  " + "-" * 66)
        for pos in closed[-10:]:
            pnl = pos.get("realized_pnl", 0)
            print(
                f"  {pos.get('signal', '?'):16s} | "
                f"PnL: ${pnl:+.2f} | "
                f"{pos.get('close_reason', '?'):12s} | "
                f"{pos.get('market_question', '?')[:40]}"
            )
        print()


def cmd_run(args):
    """Continuous trading loop."""
    global _running

    mode = "DRY RUN" if config.DRY_RUN else "LIVE TRADING"
    print(f"\n{'='*70}")
    print(f"  POLYMARKET TRADING BOT — {mode}")
    print(f"  Bankroll: ${config.BANKROLL:.2f} USDC")
    print(f"  Scan interval: {config.SCAN_INTERVAL_SECONDS}s")
    print(f"  Kelly fraction: {config.KELLY_FRACTION}")
    print(f"  Min edge: {config.MIN_EDGE:.0%}")
    print(f"  Max positions: {config.MAX_CONCURRENT_POSITIONS}")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*70}\n")

    clob_client = create_clob_client()
    cycle = 0

    while _running:
        cycle += 1
        logger.info(f"=== Trading cycle #{cycle} ===")

        try:
            # 1. Update existing positions
            state = load_state()
            state = update_position_prices(state, clob_client)
            state = check_stop_loss_take_profit(state, clob_client)

            # 2. Scan for new opportunities
            markets = scan_markets(clob_client)
            opportunities = generate_signals(markets)

            # 3. Size and execute
            portfolio = get_portfolio_state(state)
            sized = size_positions(opportunities, portfolio)

            for opp in sized:
                ok, reason = validate_trade(opp, portfolio)
                if not ok:
                    continue

                stop_loss = calculate_stop_loss(opp)
                take_profit = calculate_take_profit(opp)
                order = execute_limit_order(clob_client, opp)

                if order:
                    state = add_position(state, order, stop_loss, take_profit)
                    portfolio = get_portfolio_state(state)

            # 4. Print summary
            print(f"\n--- Cycle #{cycle} | {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')} ---")
            print(f"  Markets: {len(markets)} | Signals: {len(opportunities)} | New trades: {len(sized)}")
            print(f"  Bankroll: ${state.get('bankroll', 0):.2f} | "
                  f"Deployed: ${portfolio.total_deployed:.2f} | "
                  f"PnL: ${state.get('total_pnl', 0):+.2f}")

        except Exception as e:
            logger.error(f"Error in cycle #{cycle}: {e}", exc_info=True)

        # Wait for next cycle
        if _running:
            logger.info(f"Sleeping {config.SCAN_INTERVAL_SECONDS}s until next cycle...")
            for _ in range(config.SCAN_INTERVAL_SECONDS):
                if not _running:
                    break
                time.sleep(1)

    logger.info("Bot stopped gracefully.")
    print("\nBot stopped. Final portfolio:")
    cmd_portfolio(args)


def cmd_report(args):
    """Generate detailed performance analysis report."""
    state = load_state()

    # Try to refresh prices first
    try:
        clob_client = create_clob_client()
        state = update_position_prices(state, clob_client)
    except Exception:
        pass

    # Print the full report
    print(generate_performance_report(state))

    # Handle exports
    export_format = getattr(args, "export", None)
    if export_format:
        if export_format == "csv":
            trades_path = export_trades_csv(state)
            curve_path = export_equity_curve_csv(state)
            print(f"\n  Exported trades to:       {trades_path}")
            print(f"  Exported equity curve to: {curve_path}")
        elif export_format == "json":
            json_path = export_report_json(state)
            print(f"\n  Exported report to: {json_path}")
        elif export_format == "all":
            trades_path = export_trades_csv(state)
            curve_path = export_equity_curve_csv(state)
            json_path = export_report_json(state)
            print(f"\n  Exported trades CSV:      {trades_path}")
            print(f"  Exported equity curve:    {curve_path}")
            print(f"  Exported report JSON:     {json_path}")


def cmd_cancel_all(args):
    """Cancel all open orders."""
    clob_client = create_clob_client()
    cancel_all_orders(clob_client)
    print("All open orders cancelled.")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Polymarket Trading Strategy Bot — $99 USDC Multiplier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py scan              # Find tradeable markets
  python main.py analyze           # Full analysis with sizing
  python main.py trade             # Execute one round of trades
  python main.py run               # Continuous trading loop
  python main.py portfolio         # Check positions & P&L
  python main.py report            # Full performance report
  python main.py report --export csv   # Export trades + equity curve CSV
  python main.py report --export json  # Export full report as JSON
  python main.py report --export all   # Export everything
  python main.py cancel-all        # Emergency: cancel all orders

Configuration:
  Copy .env.example to .env and fill in your wallet details.
  Set DRY_RUN=false for live trading (default is dry run/paper trading).
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("scan", help="Scan markets for opportunities")
    subparsers.add_parser("analyze", help="Full analysis with position sizing")
    subparsers.add_parser("trade", help="Execute trades")
    subparsers.add_parser("run", help="Continuous trading loop")
    subparsers.add_parser("portfolio", help="Show portfolio status")

    report_parser = subparsers.add_parser("report", help="Performance report with P&L analysis")
    report_parser.add_argument(
        "--export", choices=["csv", "json", "all"],
        help="Export report data (csv=trades+equity curve, json=full report, all=everything)"
    )

    subparsers.add_parser("cancel-all", help="Cancel all open orders")

    args = parser.parse_args()

    commands = {
        "scan": cmd_scan,
        "analyze": cmd_analyze,
        "trade": cmd_trade,
        "run": cmd_run,
        "portfolio": cmd_portfolio,
        "report": cmd_report,
        "cancel-all": cmd_cancel_all,
    }

    if not args.command:
        parser.print_help()
        print("\n  Quick start: python main.py scan")
        sys.exit(0)

    commands[args.command](args)


if __name__ == "__main__":
    main()
