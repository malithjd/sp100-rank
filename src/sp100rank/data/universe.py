# src/sp100rank/data/universe.py
"""
Universe of 100 large-cap U.S. equities for this project.

ADR-005: see docs/decisions.md. This is a fixed, hand-curated list
of 100 large-cap names representing a stable, liquid cross-section
over 2018-2026. NOT identical to the index-defined S&P 100 — the
writeup refers to it as "100 large-cap U.S. equities" rather than
S&P 100.

Source: User-provided list (2026-04-30), with four removals:
  - FISV : Fiserv was renamed to ticker FI in 2023; keeping FISV
          creates a confusing 2018-2023 split history.
  - GOOG : Alphabet's other share class. GOOGL retained. Including
          both adds no cross-sectional information (corr > 0.999)
          while doubling Alphabet's weight in the universe.
  - EL   : Estée Lauder. Drawdown >80% during 2022-2024 with multi-
          ple gap-down events. Real history, but dominates spike
          detection and pollutes the cross-section tails.
  - PSA  : Public Storage (REIT). Dividend-heavy total return
          structure unlike the rest of the universe.

Notes on yfinance ticker formats:
  - BRK-B uses the dash form (yfinance) rather than the dot form
    (Bloomberg/Wikipedia: BRK.B).
"""

SP100_TICKERS: list[str] = [
    "AAPL",  "ABT",   "ACN",   "ADBE",  "ADP",   "AMGN",  "AMZN",  "AON",
    "APD",   "AXP",   "BA",    "BAC",   "BDX",   "BKNG",  "BLK",   "BRK-B",
    "CAT",   "CB",    "CI",    "CL",    "CMCSA", "CME",   "COST",  "CRM",
    "CSCO",  "CSX",   "CVX",   "DE",    "DHR",   "DIS",   "DUK",   "ECL",
    "EQIX",  "ETN",   "FDX",   "FIS",   "GE",    "GILD",  "GOOGL", "GS",
    "HCA",   "HD",    "HON",   "IBM",   "ICE",   "INTC",  "INTU",  "ISRG",
    "ITW",   "JNJ",   "JPM",   "KO",    "LIN",   "LLY",   "LMT",   "LOW",
    "MA",    "MCD",   "MDT",   "META",  "MRSH",   "MO",    "MRK",   "MSFT",
    "NEE",   "NFLX",  "NKE",   "NOW",   "NSC",   "NVDA",  "ORCL",  "PEP",
    "PFE",   "PG",    "PLD",   "PM",    "PNC",   "QCOM",  "REGN",  "ROP",
    "RTX",   "SCHW",  "SHW",   "SLB",   "SO",    "SPGI",  "SYK",   "TGT",
    "TJX",   "TMO",   "TXN",   "UNH",   "UPS",   "USB",   "V",     "VZ",
    "WM",    "WMT",   "XOM",   "ZTS",
]

SPX_TICKER: str = '^GSPC'

def all_tickers() -> list[str]:
    return SP100_TICKERS + [SPX_TICKER]

def is_index(ticker: str) -> bool:
    return ticker.startswith('^')

