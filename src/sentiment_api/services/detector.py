"""Detect headlines where target-entity attribution matters.

These patterns cover cases where FinBERT tends to misclassify because the
overall tone of the headline does not match the sentiment for a specific
party (plaintiff vs defendant, acquirer vs target, winner vs loser, etc.).
"""
import re


# Patterns that commonly require target-aware classification.
# Case-insensitive; matching is OR'd.
_AMBIGUOUS_PATTERNS: list[re.Pattern[str]] = [
    # Legal / litigation outcomes
    re.compile(
        r"\b(found\s+liable|jury\s+(?:found|awarded|ordered)|ordered\s+to\s+pay|"
        r"damages|disgorgement|settle(?:s|d|ment)?\b|lawsuit|sued|class\s+action|"
        r"judgment|verdict|injunction|false\s+advertising|patent\s+infringement|"
        r"copyright\s+infringement|wins?\s+(?:case|trial|appeal)|"
        r"loses?\s+(?:case|trial|appeal))\b",
        re.IGNORECASE,
    ),
    # M&A, acquisitions, tenders
    re.compile(
        r"\b(acquir(?:e|es|ed|ing|ition)|merg(?:e|es|ed|er|ing)|takeover|"
        r"tender\s+offer|buyout|go\s+private|spin[\s-]?off|divestiture|"
        r"stake\s+in|majority\s+stake)\b",
        re.IGNORECASE,
    ),
    # Partnerships / deals
    re.compile(
        r"\b(partner(?:s|ship|ed|ing)?\s+with|collab(?:oration|orate)|joint\s+venture|"
        r"strategic\s+(?:alliance|investment)|licensing\s+(?:deal|agreement)|"
        r"multi[-\s]?year\s+(?:deal|agreement)|signs?\s+(?:deal|agreement)\s+with)\b",
        re.IGNORECASE,
    ),
    # Competitive / rivalry
    re.compile(
        r"\b(rival|competitor|market\s+share|outsell|overtake|disrupt(?:s|ed)?|"
        r"threaten(?:s|ed)?\s+to|replace(?:s|d)?|kill(?:s|ed)?\s+off)\b",
        re.IGNORECASE,
    ),
    # Activist / short seller campaigns
    re.compile(
        r"\b(activist|short[\s-]?seller|short\s+report|short\s+position|"
        r"proxy\s+(?:fight|contest|battle)|board\s+(?:seat|nominee|challenge)|"
        r"hindenburg|muddy\s+waters|citron|viceroy)\b",
        re.IGNORECASE,
    ),
    # Contract / customer wins (beneficiary vs loser ambiguous)
    re.compile(
        r"\b(wins?\s+(?:contract|order|bid)|awarded\s+(?:contract|deal)|"
        r"selected\s+by|loses?\s+(?:contract|customer|account))\b",
        re.IGNORECASE,
    ),
]


# Ticker-like tokens: $AAPL or bare AAPL (2-5 uppercase letters).
# We use this as a coarse multi-symbol signal — if headline mentions multiple
# tickers, attribution often matters.
_TICKER_RE = re.compile(r"(?:\$|\b)([A-Z]{2,5})\b")

# Common uppercase tokens that are not tickers — filter out to reduce FP.
_TICKER_STOPWORDS = {
    "USA", "USD", "EU", "UK", "UN", "CEO", "CFO", "COO", "CTO", "AI", "API",
    "SEC", "FDA", "FTC", "DOJ", "IRS", "GDP", "CPI", "PPI", "IPO", "ETF",
    "NYSE", "OTC", "ADR", "EPS", "ROI", "ESG", "NEW", "YORK", "LTD", "INC",
    "CORP", "PLC", "LLC", "LP", "GMBH", "SA", "AG", "BV", "NA", "UPDATE",
    "BREAKING", "ALERT", "WATCH", "FLASH", "THE", "AND", "FOR", "PER",
    "Q1", "Q2", "Q3", "Q4", "FY", "YOY", "QOQ",
}


def extract_tickers(text: str) -> set[str]:
    """Extract ticker-like uppercase tokens, minus common stopwords."""
    if not text:
        return set()
    found = set(_TICKER_RE.findall(text))
    return found - _TICKER_STOPWORDS


def is_ambiguous(text: str, symbol: str | None = None) -> bool:
    """Return True when headline likely needs target-aware classification.

    Matches if:
      - any AMBIGUOUS_PATTERN hits, OR
      - multiple ticker-like tokens appear (beyond just the target symbol).
    """
    if not text:
        return False

    for pat in _AMBIGUOUS_PATTERNS:
        if pat.search(text):
            return True

    tickers = extract_tickers(text)
    if symbol:
        tickers.discard(symbol.upper())
    return len(tickers) >= 2
