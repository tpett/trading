"""SignalSpec + the seed SIGNALS registry (spec section 3.1).

A signal is a pure function (PanelView, as_of) -> per-symbol float score,
HIGHER = more attractive to be long. The sign convention is part of the
wrapper and is documented at each registration -- e.g. rev5 registers the
NEGATED trailing 5-day return because short-term reversal makes recent
losers attractive. NaN means "this symbol lacks the signal's inputs on this
date": the sort drops it from that date's cross-section (spec section 5.5),
never imputes.

PIT contract: fn receives a PanelView (never the raw PanelData), whose
accessors truncate at as_of -- forward data is structurally unreachable.
Price-metric semantics (trail / rvol / disthigh) are copied from
scripts/signal_scan.py so alphasearch scores agree with the scanner's panel.

The 16 seed signals (Piece 1) are followed by the 21 Tier-1 batch signals
(docs/superpowers/specs/2026-07-09-tier1-signal-batch-design.md section 2):
9 price/volume, 5 options (cp_vol/osv gated on requires_option_volume),
5 fundamentals (300-calendar-day YoY filing rule), 2 industry-relative
(the 10 frozen SEGMENTS sectors). All formulas, windows, floors, and signs
are frozen pre-registration.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading.alphasearch.panel import PanelView

SignalFn = Callable[[PanelView, pd.Timestamp], pd.Series]


@dataclass(frozen=True)
class SignalSpec:
    name: str
    fn: SignalFn
    requires_options: bool = False
    requires_fundamentals: bool = False
    # Leg volume exists only in the mid-cap gather; signals reading it are
    # refused at sweep assembly on volume-less universes (spec section 2).
    requires_option_volume: bool = False


SIGNALS: dict[str, SignalSpec] = {}


def _register(
    name: str,
    fn: SignalFn,
    *,
    requires_options: bool = False,
    requires_fundamentals: bool = False,
    requires_option_volume: bool = False,
) -> None:
    SIGNALS[name] = SignalSpec(
        name,
        fn,
        requires_options=requires_options,
        requires_fundamentals=requires_fundamentals,
        requires_option_volume=requires_option_volume,
    )


# --------------------------------------------------------------------------- #
# Price metrics -- semantics identical to scripts/signal_scan.py, but applied
# to an already-truncated close series (PanelView.closes), so the "position
# at or before the decision date" is simply the last element.
# --------------------------------------------------------------------------- #
def _trail(closes: pd.Series, window: int) -> float:
    """Trailing `window`-bar return ending at the last close."""
    p = len(closes) - 1
    if p - window < 0:
        return math.nan
    return float(closes.iloc[p] / closes.iloc[p - window] - 1)


def _rvol(closes: pd.Series, window: int = 21) -> float:
    """Annualized realized vol of the `window` bars BEFORE the last close."""
    p = len(closes) - 1
    if p < window:
        return math.nan
    return float(closes.iloc[p - window : p].pct_change().std() * math.sqrt(252))


def _disthigh(closes: pd.Series, window: int = 252) -> float:
    """Distance from the trailing `window`-bar high (<= 0; 0 = at the high)."""
    p = len(closes) - 1
    if p < 0:
        return math.nan
    return float(closes.iloc[p] / closes.iloc[max(0, p - window) : p + 1].max() - 1)


def _price_signal(metric: Callable[[pd.Series], float]) -> SignalFn:
    def fn(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
        scores = {symbol: metric(view.closes(symbol)) for symbol in view.symbols}
        return pd.Series(scores, dtype="float64")

    return fn


def _mom(window: int) -> SignalFn:
    def metric(closes: pd.Series, _window: int = window) -> float:
        return _trail(closes, _window)

    return _price_signal(metric)


# Price family. Direction rationale recorded per registration (spec 3.1).
_register("mom21", _mom(21))  # momentum: recent winners attractive
_register("mom63", _mom(63))
_register("mom126", _mom(126))
_register("mom252", _mom(252))
# Short-term reversal: recent LOSERS attractive -> negate the trailing return.
_register("rev5", _price_signal(lambda c: -_trail(c, 5)))
# Low-vol anomaly: quiet names attractive -> negate realized vol.
_register("rvol21", _price_signal(lambda c: -_rvol(c)))
# Proximity to the 52-week high is momentum-like; disthigh is <= 0 with 0 at
# the high, so raw sign already puts near-high names on top.
_register("disthigh", _price_signal(_disthigh))


# --------------------------------------------------------------------------- #
# Tier-1 price/volume family (spec 2026-07-09 section 2). Bar metrics receive
# the PIT-truncated BAR_COLUMNS frame; the last row IS the as_of bar (the
# _trail convention). Windows/floors/signs are frozen pre-registration.
# --------------------------------------------------------------------------- #
_PARKINSON_DENOM = 4.0 * math.log(2.0)


def _bar_signal(metric: Callable[[pd.DataFrame], float]) -> SignalFn:
    def fn(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
        scores = {symbol: metric(view.bars(symbol)) for symbol in view.symbols}
        return pd.Series(scores, dtype="float64")

    return fn


def _feature_signal(column: str, sign: float) -> SignalFn:
    """Score = sign * the precomputed rolling feature gathered as-of."""

    def fn(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
        scores = {s: sign * view.feature(s, column) for s in view.symbols}
        return pd.Series(scores, dtype="float64")

    return fn


def _mom_12_2(closes: pd.Series) -> float:
    """Total return t-252 -> t-21 (skip the most recent month)."""
    p = len(closes) - 1
    if p - 252 < 0:
        return math.nan
    return float(closes.iloc[p - 21] / closes.iloc[p - 252] - 1)


def _overnight(bars: pd.DataFrame) -> float:
    """Sum of the last 63 overnight log-returns ln(open_t / close_{t-1})."""
    if len(bars) < 64:
        return math.nan
    opens = bars["open"].to_numpy()[-63:]
    prev_close = bars["close"].to_numpy()[-64:-1]
    return float(np.sum(np.log(opens / prev_close)))


def _park_vol(bars: pd.DataFrame) -> float:
    """Parkinson range vol over 21 bars: sqrt(mean(ln(H/L)^2/(4 ln 2))) * sqrt(252)."""
    if len(bars) < 21:
        return math.nan
    high = bars["high"].to_numpy()[-21:]
    low = bars["low"].to_numpy()[-21:]
    terms = np.log(high / low) ** 2 / _PARKINSON_DENOM
    return float(math.sqrt(terms.mean()) * math.sqrt(252))


def _max5(closes: pd.Series) -> float:
    """Mean of the 5 largest daily returns among the last 21."""
    if len(closes) < 22:
        return math.nan
    rets = closes.iloc[-22:].pct_change().dropna().to_numpy()
    return float(np.sort(rets)[-5:].mean())


def _amihud(bars: pd.DataFrame) -> float:
    """Mean |ret| / dollar volume over the last 252 bars; min 126 valid terms
    (non-positive dollar volume or NaN return terms are skipped, never 0)."""
    window = bars.iloc[-252:]
    rets = window["close"].pct_change().to_numpy()
    dollar = (window["close"] * window["volume"]).to_numpy()
    valid = ~np.isnan(rets) & ~np.isnan(dollar) & (dollar > 0)
    if valid.sum() < 126:
        return math.nan
    return float(np.mean(np.abs(rets[valid]) / dollar[valid]))


def _vol_trend(bars: pd.DataFrame) -> float:
    """Mean dollar volume over 21 bars / mean over 252 bars (strict window)."""
    if len(bars) < 252:
        return math.nan
    dollar = (bars["close"] * bars["volume"]).to_numpy()[-252:]
    base = dollar.mean()
    if not base > 0:  # NaN or zero baseline both land here
        return math.nan
    return float(dollar[-21:].mean() / base)


def _div_yield(bars: pd.DataFrame) -> float:
    """Trailing 252-bar cash dividends / last close. min_count=1 keeps an
    all-NaN div_cash column (legacy narrow cache) NaN instead of sum()'s
    skipna zero -- a cache without dividend data must not claim 'no
    dividends' (that would fabricate a 0-yield cross-section)."""
    if len(bars) < 252:
        return math.nan
    paid = bars["div_cash"].iloc[-252:].sum(min_count=1)
    close = float(bars["close"].iloc[-1])
    if pd.isna(paid) or not close > 0:
        return math.nan
    return float(paid / close)


# Classic UMD with the skip-month that avoids short-term reversal
# contamination (Jegadeesh-Titman).
_register("mom_12_2", _price_signal(_mom_12_2))
# The overnight component of returns persists (Lou-Polk-Skouras).
_register("overnight", _bar_signal(_overnight))
# Low-vol anomaly on the Parkinson range estimator -> negate.
_register("park_vol", _bar_signal(lambda b: -_park_vol(b)))
# Idiosyncratic-vol puzzle (Ang-Hodrick-Xing-Zhang) -> negate.
_register("ivol", _feature_signal("ivol", -1.0))
# Lottery demand: extreme-day chasers overpay (Bali-Cakici-Whitelaw) -> negate.
_register("max5", _price_signal(lambda c: -_max5(c)))
# Betting-against-beta (Frazzini-Pedersen) -> negate.
_register("beta", _feature_signal("beta", -1.0))
# Illiquidity premium (Amihud): harder-to-trade names pay more.
_register("amihud", _bar_signal(_amihud))
# High-volume return premium (Gervais-Kaniel-Mingelgrin).
_register("vol_trend", _bar_signal(_vol_trend))
# Income/value tilt: cash actually paid out over the trailing year.
_register("div_yield", _bar_signal(_div_yield))


# --------------------------------------------------------------------------- #
# Options family (requires_options). Column values come straight from
# panel.cell_metrics; each registration bakes in the higher-=-attractive sign.
# --------------------------------------------------------------------------- #
def _option_signal(column: str, sign: float) -> SignalFn:
    def fn(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
        scores: dict[str, float] = {}
        for symbol in view.symbols:
            row = view.option_row(symbol)
            scores[symbol] = math.nan if row is None else sign * float(row[column])
        return pd.Series(scores, dtype="float64")

    return fn


def _vrp(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
    """Volatility risk premium: ATM implied vol minus trailing realized vol
    (exactly signal_scan's vrp = atm_iv - rvol21)."""
    scores: dict[str, float] = {}
    for symbol in view.symbols:
        row = view.option_row(symbol)
        if row is None:
            scores[symbol] = math.nan
            continue
        scores[symbol] = float(row["atm_iv"]) - _rvol(view.closes(symbol))
    return pd.Series(scores, dtype="float64")


# Rich vol premium = overpaid insurance on the name -> attractive to own.
_register("vrp", _vrp, requires_options=True)
# Spec 3.1's example: the skew signal registers NEGATED (-skew_put_atm), so
# LESS downside-hedged names score higher -- the risk-veto intuition.
_register("hedge", _option_signal("hedge", -1.0), requires_options=True)
# cell excite is already -skew_put_call (call-side richness): keep raw sign.
_register("excite", _option_signal("excite", +1.0), requires_options=True)
# Lottery/vol-premium effect: high-IV names underperform -> negate.
_register("atm_iv", _option_signal("atm_iv", -1.0), requires_options=True)
# Convex smile = tail fear priced in -> negate.
_register("smile", _option_signal("smile", -1.0), requires_options=True)
# Option-illiquidity premium: WIDER spread predicted higher returns in the
# mid-cap scan (docs/experiments.md section 9) -> keep raw sign.
_register("atm_spread", _option_signal("atm_spread", +1.0), requires_options=True)


# --------------------------------------------------------------------------- #
# Tier-1 options family (spec 2026-07-09 section 2). cp_vol/osv read per-leg
# volume, which only the mid-cap gather carries -> requires_option_volume
# refuses them elsewhere (fake log(1/1)=0 cross-sections must never trial).
# --------------------------------------------------------------------------- #
def _osv(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
    """Option/stock dollar-volume ratio: the cell's opt_dollar_vol over the
    decision-day stock dollar volume (close*volume of the last bar <= as_of),
    one cell vs one day. NEGATED at registration below."""
    scores: dict[str, float] = {}
    for symbol in view.symbols:
        row = view.option_row(symbol)
        score = math.nan
        if row is not None:
            bars = view.bars(symbol)
            if len(bars):
                stock_dollar = float(bars["close"].iloc[-1] * bars["volume"].iloc[-1])
                opt_dollar = float(row["opt_dollar_vol"])
                if stock_dollar > 0 and not math.isnan(opt_dollar):
                    score = -(opt_dollar / stock_dollar)
        scores[symbol] = score
    return pd.Series(scores, dtype="float64")


def _option_innovation(column: str, sign: float) -> SignalFn:
    """sign * (current cell's column - prior cell's column); NaN when either
    cell is missing or the prior is stale (option_row_prior's 45d cap)."""

    def fn(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
        scores: dict[str, float] = {}
        for symbol in view.symbols:
            row = view.option_row(symbol)
            prior = view.option_row_prior(symbol)
            if row is None or prior is None:
                scores[symbol] = math.nan
            else:
                scores[symbol] = sign * (float(row[column]) - float(prior[column]))
        return pd.Series(scores, dtype="float64")

    return fn


# Informed call demand predicts positive returns (Pan-Poteshman). The cell's
# committed cp_vol column is log(1+call volume) - log(1+put volume) with the
# ATM leg counted as the call it is.
_register("cp_vol", _option_signal("cp_vol", +1.0),
          requires_options=True, requires_option_volume=True)
# High option/stock volume marks informed (mostly bearish) positioning
# (Johnson-So) -> negate.
_register("osv", _osv, requires_options=True, requires_option_volume=True)
# Steep OTM-put smirk predicts negative returns (Xing-Zhang-Zhao) -> negate.
_register("otm_put_iv", _option_signal("otm_put_iv", -1.0), requires_options=True)
# Rising implied vol = rising perceived risk (An-Ang-Bali-Cakici) -> negate.
_register("iv_change", _option_innovation("atm_iv", -1.0), requires_options=True)
# A steepening put smirk is bearish, consistent with `hedge`'s sign -> negate.
_register("dskew", _option_innovation("hedge", -1.0), requires_options=True)


# --------------------------------------------------------------------------- #
# Fundamentals family (requires_fundamentals). Ratios computed at scoring time
# from price-free stored primitives x the as-of close, exactly as
# trading.signals.quality / trading.signals.value do. NaN (dropped) when no
# filing is visible -- the sort's missing-data rule handles it; there is no
# 0.5-neutral here because a cross-sectional SORT needs a real value.
# --------------------------------------------------------------------------- #
def _fundamental_field(view: PanelView, symbol: str, key: str) -> float:
    row = view.fundamentals_row(symbol)
    if row is None or key not in row.index:
        return math.nan
    return float(row[key])


def _gross_profitability(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
    scores = {s: _fundamental_field(view, s, "gross_profitability") for s in view.symbols}
    return pd.Series(scores, dtype="float64")


def _value_ratio(numerator: str) -> SignalFn:
    def fn(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
        scores: dict[str, float] = {}
        for symbol in view.symbols:
            shares = _fundamental_field(view, symbol, "shares_outstanding")
            value = _fundamental_field(view, symbol, numerator)
            close = view.last_close(symbol)
            score = math.nan
            if not math.isnan(shares) and shares > 0 and not math.isnan(close):
                market_cap = shares * close
                if market_cap > 0 and not math.isnan(value):
                    # Negative income/equity is a real (bottom-ranked) value,
                    # not missing data -- same policy as trading.signals.value.
                    score = value / market_cap
            scores[symbol] = score
        return pd.Series(scores, dtype="float64")

    return fn


_register("gross_profitability", _gross_profitability, requires_fundamentals=True)
_register("earnings_yield", _value_ratio("ttm_net_income"), requires_fundamentals=True)
_register("book_to_market", _value_ratio("book_equity"), requires_fundamentals=True)


# --------------------------------------------------------------------------- #
# Tier-1 fundamentals family (spec 2026-07-09 section 2). YoY = latest filing
# vs the latest filing FILED >= 300 calendar days earlier
# (PanelView.fundamentals_row_prior); missing/ineligible -> NaN, dropped.
# --------------------------------------------------------------------------- #
def _fund_value(row: pd.Series | None, key: str) -> float:
    if row is None or key not in row.index:
        return math.nan
    return float(row[key])


def _yoy_growth(key: str, sign: float) -> SignalFn:
    """sign * (current/prior - 1) of one stored primitive; NaN unless both
    filings carry a value and the prior is strictly positive (a ratio against
    a non-positive base has no growth interpretation)."""

    def fn(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
        scores: dict[str, float] = {}
        for symbol in view.symbols:
            now = _fund_value(view.fundamentals_row(symbol), key)
            then = _fund_value(view.fundamentals_row_prior(symbol), key)
            score = math.nan
            if not math.isnan(now) and not math.isnan(then) and then > 0:
                score = sign * (now / then - 1.0)
            scores[symbol] = score
        return pd.Series(scores, dtype="float64")

    return fn


def _roa_of(row: pd.Series | None) -> float:
    ni = _fund_value(row, "ttm_net_income")
    assets = _fund_value(row, "assets")
    if math.isnan(ni) or not assets > 0:
        return math.nan
    return ni / assets


def _roa(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
    scores = {s: _roa_of(view.fundamentals_row(s)) for s in view.symbols}
    return pd.Series(scores, dtype="float64")


def _droa(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
    scores = {
        s: _roa_of(view.fundamentals_row(s)) - _roa_of(view.fundamentals_row_prior(s))
        for s in view.symbols
    }  # NaN propagates from either leg
    return pd.Series(scores, dtype="float64")


def _net_issuance(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
    """Split-adjusted shares_outstanding YoY, NEGATED (issuers underperform,
    Pontiff-Woodgate). Comparable prior shares = prior * product of
    split_factor over bar dates in (prior_filed, current_filed]. ANY NaN
    split_factor in that window -> NaN: a legacy narrow cache cannot claim
    "no splits", and prod()'s skipna would fabricate exactly that."""
    scores: dict[str, float] = {}
    for symbol in view.symbols:
        current = view.fundamentals_row(symbol)
        prior = view.fundamentals_row_prior(symbol)
        score = math.nan
        if current is not None and prior is not None:
            shares_now = _fund_value(current, "shares_outstanding")
            shares_then = _fund_value(prior, "shares_outstanding")
            if not math.isnan(shares_now) and shares_then > 0:
                factors = view.bars(symbol)["split_factor"]
                window = factors[(factors.index > prior.name)
                                 & (factors.index <= current.name)]
                if not window.isna().any():
                    adjustment = float(window.prod()) if len(window) else 1.0
                    if adjustment > 0:
                        score = -(shares_now / (shares_then * adjustment) - 1.0)
        scores[symbol] = score
    return pd.Series(scores, dtype="float64")


# Investment factor: asset growers underperform (Cooper-Gulen-Schill) -> negate.
_register("asset_growth", _yoy_growth("assets", -1.0), requires_fundamentals=True)
# Issuance anomaly (Pontiff-Woodgate): negation lives inside _net_issuance.
_register("net_issuance", _net_issuance, requires_fundamentals=True)
# Quality: profitable-per-asset names outperform.
_register("roa", _roa, requires_fundamentals=True)
# Fundamental momentum: improving profitability.
_register("droa", _droa, requires_fundamentals=True)
# Growth: rising trailing revenue.
_register("rev_growth", _yoy_growth("revenue_ttm", +1.0), requires_fundamentals=True)


# --------------------------------------------------------------------------- #
# Tier-1 industry-relative family (spec 2026-07-09 section 2): the 10 frozen
# SEGMENTS sectors are the industry partition (via the committed sic_map,
# threaded onto the panel as PanelData.sectors). Sector stats come ONLY from
# symbols present in the date's cross-section; unmapped symbols score NaN.
# --------------------------------------------------------------------------- #
def _sector_means(view: PanelView, values: dict[str, float]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for symbol in view.symbols:
        sector = view.sector(symbol)
        value = values[symbol]
        if sector is None or math.isnan(value):
            continue
        sums[sector] = sums.get(sector, 0.0) + value
        counts[sector] = counts.get(sector, 0) + 1
    return {sector: sums[sector] / counts[sector] for sector in sums}


def _ind_mom(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
    values = {s: _mom_12_2(view.closes(s)) for s in view.symbols}
    means = _sector_means(view, values)
    scores = {
        s: (means.get(sector, math.nan)
            if (sector := view.sector(s)) is not None else math.nan)
        for s in view.symbols
    }
    return pd.Series(scores, dtype="float64")


def _ind_rel_rev(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
    values = {s: _trail(view.closes(s), 21) for s in view.symbols}
    means = _sector_means(view, values)
    scores: dict[str, float] = {}
    for symbol in view.symbols:
        sector = view.sector(symbol)
        own = values[symbol]
        if sector is None or sector not in means or math.isnan(own):
            scores[symbol] = math.nan
        else:
            scores[symbol] = -(own - means[sector])
    return pd.Series(scores, dtype="float64")


# Industry momentum (Moskowitz-Grinblatt): a hot sector lifts every member.
_register("ind_mom", _ind_mom)
# Within-industry reversal (Da-Liu-Schaumburg, 21d at monthly cadence):
# laggards vs their sector mean recover -- the minus lives IN the formula,
# so registration is raw (spec section 2 defines the signal WITH the minus).
_register("ind_rel_rev", _ind_rel_rev)
