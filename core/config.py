MAX_SCORE = 10.0
MIN_SCORE = 0.0
DEFAULT_ACTION = "ANALYSIS_ONLY"
DEFAULT_CONFIDENCE = 0.5

MARKET_FEATURE_VERSION = "market-feature-v1"
CURRENT_SCORE_VERSION = "current-score-v2"
LONG_TERM_SCORE_VERSION = "long-term-score-v2"
NEWS_CONTRACT_VERSION = "news-contract-v1"

# Evidence-based confidence model (see core.score_engine._compute_confidence).
# Each factor produces a value in [0, 1]; the confidence is the weighted sum
# minus explicit risk penalties, clamped to [CONFIDENCE_FLOOR, CONFIDENCE_CAP].
# Weights sum to 1.0 so the baseline stays interpretable as a percentage.
CONFIDENCE_VERSION = "evidence-confidence-v2"

CONFIDENCE_WEIGHT_DATA_QUALITY = 0.25
CONFIDENCE_WEIGHT_SOURCE_RELIABILITY = 0.20
CONFIDENCE_WEIGHT_SIGNAL_AGREEMENT = 0.20
CONFIDENCE_WEIGHT_FRESHNESS = 0.15
CONFIDENCE_WEIGHT_HISTORY_COVERAGE = 0.10
CONFIDENCE_WEIGHT_VOLATILITY_REGIME = 0.10

# Calendar-age gap between the newest bar used and as_of before freshness decays,
# and how many calendar days after that grace window reach zero freshness credit.
CONFIDENCE_FRESHNESS_GRACE_DAYS = 4
CONFIDENCE_FRESHNESS_DECAY_DAYS = 26

# Minimum number of valid daily bars for full long-window (200d MA) coverage.
CONFIDENCE_FULL_COVERAGE_BARS = 230

# Daily-return standard deviation treated as fully calm versus fully chaotic.
CONFIDENCE_VOL_CALM_DAILY_STD = 0.015
CONFIDENCE_VOL_CHAOTIC_DAILY_STD = 0.060

# Dead-zone half-width used when reading factor directions so tiny moves do not
# flip a signal between bullish/bearish arbitrarily.
CONFIDENCE_SIGNAL_DEADZONE_RATIO = 0.002

# Named penalties applied once per condition, replacing the previous flat -0.20
# per category. Scaled to reflect how much each condition undermines the result.
RISK_FLAG_CONFIDENCE_PENALTIES = {
    "Weak momentum": 0.10,
    "Low volume": 0.08,
    "Downtrend": 0.12,
    "RSI extreme": 0.06,
}
FUNDAMENTAL_SOURCE_PENALTY = 0.10
GOVERNANCE_RISK_GATE_PENALTY = 0.05

CONFIDENCE_FLOOR = 0.10
CONFIDENCE_CAP = 0.95
