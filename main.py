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
    python main.py risk          # Sharpe, Sortino, VaR, drawdown analysis
    python main.py intel         # Market correlations, whales, momentum
    python main.py backtest      # Backtest strategies on price history
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
from market_scanner import scan_markets, MarketInfo, test_connectivity
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
from risk_metrics import compute_all_risk_metrics, format_risk_report
from execution_tracker import compute_execution_metrics, format_execution_report
from price_tracker import (
    get_tracked_markets,
    compute_market_analytics,
    backtest_strategy,
    mean_reversion_strategy,
    momentum_strategy,
    volume_spike_strategy,
    format_backtest_report,
)
from market_intelligence import format_market_intelligence_report

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

    # Risk metrics
    risk = compute_all_risk_metrics(state)
    print(format_risk_report(risk))

    # Execution quality
    exec_metrics = compute_execution_metrics()
    print(format_execution_report(exec_metrics))

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


def cmd_risk(args):
    """Display advanced risk metrics."""
    state = load_state()
    risk = compute_all_risk_metrics(state)

    print(f"\n{'='*80}")
    print("  RISK DASHBOARD")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*80}")
    print(format_risk_report(risk))
    print(format_execution_report(compute_execution_metrics()))
    print(f"\n{'='*80}")


def cmd_intel(args):
    """Display market intelligence: correlations, whales, category momentum."""
    logger.info("Gathering market intelligence...")

    clob_client = None
    try:
        clob_client = create_clob_client()
    except Exception:
        logger.warning("Running without CLOB client")

    markets = scan_markets(clob_client)

    print(f"\n{'='*80}")
    print(f"  MARKET INTELLIGENCE — {len(markets)} markets analyzed")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*80}")
    print(format_market_intelligence_report(markets, clob_client))

    # Show time-to-resolution summary
    print("\n  TIME-TO-RESOLUTION SUMMARY:")
    print("  " + "-" * 76)
    regime_counts = {}
    for m in markets:
        r = m.theta_regime
        regime_counts[r] = regime_counts.get(r, 0) + 1

    for regime, count in sorted(regime_counts.items()):
        print(f"    {regime:20s}  {count:>4d} markets")

    print(f"\n{'='*80}")


def cmd_backtest(args):
    """Backtest strategies against historical price data."""
    tracked = get_tracked_markets()

    if not tracked:
        print("\nNo price history available yet. Run 'python main.py scan' a few times first")
        print("to build up price snapshots, then backtest against them.")
        return

    market_id = getattr(args, "market_id", None)

    strategies = {
        "mean_reversion": mean_reversion_strategy,
        "momentum": momentum_strategy,
        "volume_spike": volume_spike_strategy,
    }
    strategy_name = getattr(args, "strategy", "mean_reversion")

    if strategy_name not in strategies:
        print(f"Unknown strategy: {strategy_name}. Available: {', '.join(strategies.keys())}")
        return

    strategy_fn = strategies[strategy_name]

    print(f"\n{'='*80}")
    print(f"  BACKTESTING — Strategy: {strategy_name}")
    print(f"{'='*80}")

    if market_id:
        # Backtest single market
        question = tracked.get(market_id, market_id)
        result = backtest_strategy(market_id, strategy_fn)
        print(format_backtest_report(result, question))
    else:
        # Backtest all tracked markets
        results = []
        for mid, question in tracked.items():
            analytics = compute_market_analytics(mid)
            if analytics.get("snapshots", 0) < 10:
                continue
            result = backtest_strategy(mid, strategy_fn)
            if result.get("status") == "ok":
                result["market_id"] = mid
                result["question"] = question
                results.append(result)

        if not results:
            print("\n  No markets have enough price history for backtesting (need >= 10 snapshots).")
            print(f"  Tracked markets: {len(tracked)}")
            return

        # Summary table
        table_data = []
        for r in sorted(results, key=lambda x: x["total_return_pct"], reverse=True):
            table_data.append([
                r["question"][:40] + "...",
                r["total_trades"],
                f"{r['win_rate']:.0f}%",
                f"${r['total_return']:+.2f}",
                f"{r['total_return_pct']:+.1f}%",
                f"{r['max_drawdown_pct']:.1f}%",
                f"{r['sharpe']:.3f}",
                r["snapshots_used"],
            ])

        headers = ["Market", "Trades", "Win%", "Return$", "Return%", "MaxDD", "Sharpe", "Data"]
        print(f"\n  Results across {len(results)} markets:")
        print("  " + tabulate(table_data, headers=headers, tablefmt="simple").replace("\n", "\n  "))

        # Aggregate stats
        total_return = sum(r["total_return"] for r in results)
        avg_sharpe = sum(r["sharpe"] for r in results) / len(results)
        avg_win_rate = sum(r["win_rate"] for r in results) / len(results)
        profitable = len([r for r in results if r["total_return"] > 0])

        print(f"\n  Aggregate: {profitable}/{len(results)} profitable | "
              f"Total: ${total_return:+.2f} | "
              f"Avg Sharpe: {avg_sharpe:.3f} | "
              f"Avg Win Rate: {avg_win_rate:.0f}%")

    # Show price analytics if single market
    if market_id:
        analytics = compute_market_analytics(market_id)
        if analytics.get("status") == "ok":
            print(f"\n  PRICE ANALYTICS:")
            print(f"    Current:     {analytics['current_price']:.4f}")
            print(f"    Trend:       {analytics['trend_direction']} ({analytics['trend_per_hour']:+.6f}/hr)")
            print(f"    Volatility:  {analytics['volatility']:.4f}")
            print(f"    SMA 5/20:    {analytics['sma_5']:.4f} / {analytics['sma_20']:.4f} ({analytics['sma_crossover']})")
            print(f"    Support:     {analytics['support']:.4f}")
            print(f"    Resistance:  {analytics['resistance']:.4f}")
            print(f"    Volume:      {analytics['volume_ratio']:.1f}x average")

    print(f"\n{'='*80}")


def cmd_test(args):
    """Test connectivity to Polymarket APIs."""
    print(f"\n{'='*70}")
    print("  CONNECTIVITY DIAGNOSTICS")
    print(f"{'='*70}\n")

    results = test_connectivity()

    # Internet check
    inet = results.get("internet", {})
    status = "OK" if inet.get("ok") else "FAIL"
    print(f"  Internet:     [{status}]  Your IP: {inet.get('your_ip', '?')}")

    # Gamma API
    gamma = results.get("gamma_api", {})
    status = "OK" if gamma.get("ok") else "FAIL"
    print(f"  Gamma API:    [{status}]  HTTP {gamma.get('status', '?')}"
          f"  Server: {gamma.get('server', '?')}")
    if gamma.get("error"):
        print(f"                Error: {gamma['error']}")
    if gamma.get("ok") and gamma.get("body_preview"):
        # Check if we got JSON or a Cloudflare page
        preview = gamma["body_preview"]
        if preview.startswith("[") or preview.startswith("{"):
            print(f"                Response: Valid JSON")
        else:
            print(f"                Response: {preview[:100]}...")

    # CLOB API
    clob = results.get("clob_api", {})
    status = "OK" if clob.get("ok") else "FAIL"
    print(f"  CLOB API:     [{status}]  HTTP {clob.get('status', '?')}"
          f"  Server: {clob.get('server', '?')}")
    if clob.get("error"):
        print(f"                Error: {clob['error']}")

    # CLOB time
    clob_time = results.get("clob_time", {})
    status = "OK" if clob_time.get("ok") else "FAIL"
    print(f"  CLOB /time:   [{status}]  {clob_time.get('body', clob_time.get('error', '?'))}")

    # CLOB client auth
    print()
    if config.PRIVATE_KEY:
        print("  Wallet:       Private key configured")
        print(f"  Address:      {config.WALLET_ADDRESS or 'not set'}")
        print(f"  Sig type:     {config.SIGNATURE_TYPE}")
        try:
            client = create_clob_client()
            print("  Auth:         [OK] CLOB client authenticated")
        except Exception as e:
            print(f"  Auth:         [FAIL] {e}")
    else:
        print("  Wallet:       No PRIVATE_KEY set (read-only mode)")

    print(f"\n  Config:")
    print(f"    Gamma URL:  {config.GAMMA_API_URL}")
    print(f"    CLOB URL:   {config.CLOB_API_URL}")
    print(f"    Chain ID:   {config.CHAIN_ID}")
    print(f"    Bankroll:   ${config.BANKROLL:.2f}")
    print(f"    Dry run:    {config.DRY_RUN}")

    # Recommendations
    print()
    all_ok = gamma.get("ok") and clob.get("ok")
    if all_ok:
        print("  All APIs reachable. You're good to trade!")
    elif inet.get("ok"):
        print("  TROUBLESHOOTING:")
        print("  Your internet works but Polymarket APIs are blocked.")
        print("  Possible causes:")
        print("    1. Cloudflare is blocking your IP (try a different VPN region)")
        print("    2. Your ISP/network blocks these domains")
        print("    3. Polymarket is geo-restricted in your region")
        print()
        print("  Try these VPN regions: Netherlands, Germany, UK, Singapore")
        print("  Avoid: US (some restrictions), Japan (some datacenter IPs blocked)")
    else:
        print("  No internet connectivity detected. Check your network.")

    print(f"\n{'='*70}")


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
  python main.py report            # Full performance report with risk metrics
  python main.py report --export all   # Export trades CSV + JSON + equity curve
  python main.py risk              # Sharpe, Sortino, VaR, drawdown dashboard
  python main.py intel             # Cross-market correlations & whale signals
  python main.py backtest          # Backtest all tracked markets
  python main.py backtest --strategy momentum --market-id <id>
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

    subparsers.add_parser("risk", help="Advanced risk metrics (Sharpe, VaR, drawdown)")
    subparsers.add_parser("intel", help="Market intelligence (correlations, whales, momentum)")

    bt_parser = subparsers.add_parser("backtest", help="Backtest strategies against price history")
    bt_parser.add_argument("--market-id", help="Backtest a specific market (default: all tracked)")
    bt_parser.add_argument("--strategy", default="mean_reversion",
                           choices=["mean_reversion", "momentum", "volume_spike"],
                           help="Strategy to backtest (default: mean_reversion)")

    subparsers.add_parser("test", help="Test connectivity to Polymarket APIs")
    subparsers.add_parser("cancel-all", help="Cancel all open orders")

    args = parser.parse_args()

    commands = {
        "scan": cmd_scan,
        "analyze": cmd_analyze,
        "trade": cmd_trade,
        "run": cmd_run,
        "portfolio": cmd_portfolio,
        "report": cmd_report,
        "risk": cmd_risk,
        "intel": cmd_intel,
        "backtest": cmd_backtest,
        "test": cmd_test,
        "cancel-all": cmd_cancel_all,
    }

    if not args.command:
        parser.print_help()
        print("\n  Quick start: python main.py scan")
        sys.exit(0)

    commands[args.command](args)


if __name__ == "__main__":
    main()
