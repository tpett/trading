"""Black-Scholes implied volatility and IV-skew helpers for the options POC.

Scope
-----
This module inverts European Black-Scholes prices for implied volatility and,
from a small gathered sample of contracts around the money, derives two crude
skew measures. It is a proof-of-concept: we want to know whether IV skew
carries any cross-sectional signal about forward stock returns before paying a
vendor for a real options surface.

POC approximations (documented, deliberate)
--------------------------------------------
* ``RATE`` -- a flat 4.5% continuously-compounded risk-free rate. A real build
  would use a term-structure (e.g. the OIS/Treasury curve interpolated to each
  expiry). For 20-45 DTE contracts the level of the short rate barely moves the
  inverted vol, so a constant is fine for a signal-existence test.
* ``DIV_YIELD`` -- a flat 0% dividend yield. Most liquid single names pay
  little over a sub-two-month horizon; ignoring dividends biases inverted vols
  by a negligible amount relative to the bid/ask noise in a single print. A
  real build would carry a per-name forward dividend estimate.

Both are module-level constants so the assumption is stated in exactly one
place and is easy to revisit.

Everything here stays in the standard library + math; the normal CDF is built
from ``math.erf`` and the inversion is done by hand (bisection with a Newton
step), so the module adds no dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --- POC constants (see module docstring) ---------------------------------
RATE = 0.045
DIV_YIELD = 0.0

# An option close at or below this is treated as a placeholder / too illiquid
# to invert (Robinhood's gap-filled expired bars park at $0.01). Legs at or
# below it are dropped in skew_from_cell alongside interpolated=True legs.
_MIN_OPTION_CLOSE = 0.03

# --- Inversion tuning ------------------------------------------------------
# Sigma is bracketed here. 1e-4 is a floor (below it the price is
# indistinguishable from intrinsic); 5.0 (500% vol) is far above anything a
# real equity option quotes at, so it safely brackets the root.
_SIGMA_LOW = 1e-4
_SIGMA_HIGH = 5.0

# A print at or below this dollar amount carries no reliable vol information:
# deep-OTM options tick in pennies and a single unreliable print inverts to
# garbage. We also require at least this much *time value* (price above the
# no-arbitrage intrinsic floor) before attempting an inversion.
_MIN_PRICE = 1e-3

# Bisection stops once the sigma bracket is this narrow. 1e-8 is far tighter
# than the 1e-3 round-trip tolerance we advertise, leaving comfortable margin.
_SIGMA_TOL = 1e-8
_MAX_ITERS = 200


def norm_cdf(x: float) -> float:
    """Standard normal CDF via ``math.erf`` (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_price(
    spot: float,
    strike: float,
    t_years: float,
    rate: float,
    div_yield: float,
    sigma: float,
    is_call: bool,
) -> float:
    """European Black-Scholes price. This is the function we invert.

    Degenerate inputs (non-positive time or vol) collapse to the discounted
    intrinsic payoff, which keeps the inversion's bracket endpoints well
    defined.
    """
    disc_spot = spot * math.exp(-div_yield * t_years)
    disc_strike = strike * math.exp(-rate * t_years)
    if t_years <= 0.0 or sigma <= 0.0:
        if is_call:
            return max(disc_spot - disc_strike, 0.0)
        return max(disc_strike - disc_spot, 0.0)

    vol_sqrt_t = sigma * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (rate - div_yield + 0.5 * sigma * sigma) * t_years) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    if is_call:
        return disc_spot * norm_cdf(d1) - disc_strike * norm_cdf(d2)
    return disc_strike * norm_cdf(-d2) - disc_spot * norm_cdf(-d1)


def implied_vol(
    price: float,
    spot: float,
    strike: float,
    t_years: float,
    rate: float = RATE,
    div_yield: float = DIV_YIELD,
    is_call: bool = True,
) -> float | None:
    """Invert the BS European price for sigma.

    Returns ``None`` (rather than a bogus number) when the quote cannot be
    inverted stably:

    * non-finite / non-positive inputs;
    * a price below the no-arbitrage intrinsic floor (an arbitrageable or
      stale print);
    * a near-zero price or one that is essentially all intrinsic value -- there
      is no vol information in the remaining pennies, and deep-OTM near-$0
      prints are exactly the unreliable ticks we must guard against;
    * a price above the theoretical upper bound (would require sigma beyond the
      bracket).

    Otherwise it brackets sigma in ``[_SIGMA_LOW, _SIGMA_HIGH]`` and solves by
    bisection, taking a Newton (vega) step whenever that step stays inside the
    current bracket. Bisection guarantees convergence; Newton just accelerates.
    """
    if price is None or not math.isfinite(price):
        return None
    if spot <= 0.0 or strike <= 0.0 or t_years <= 0.0:
        return None
    if price < _MIN_PRICE:
        return None

    disc_spot = spot * math.exp(-div_yield * t_years)
    disc_strike = strike * math.exp(-rate * t_years)
    intrinsic = max(disc_spot - disc_strike, 0.0) if is_call else max(disc_strike - disc_spot, 0.0)

    # Below the no-arbitrage floor: not invertible.
    if price < intrinsic - 1e-9:
        return None
    # Essentially all intrinsic -> no vol content to extract.
    if price - intrinsic < _MIN_PRICE:
        return None

    def priced(sigma: float) -> float:
        return bs_price(spot, strike, t_years, rate, div_yield, sigma, is_call)

    lo, hi = _SIGMA_LOW, _SIGMA_HIGH
    f_lo = priced(lo) - price
    f_hi = priced(hi) - price
    # Price is monotone increasing in sigma. If the target lies outside the
    # bracketed price range it cannot be inverted within [low, high].
    if f_lo > 0.0 or f_hi < 0.0:
        return None

    sigma = 0.5 * (lo + hi)
    for _ in range(_MAX_ITERS):
        f = priced(sigma) - price
        if f > 0.0:
            hi = sigma
        else:
            lo = sigma
        if hi - lo < _SIGMA_TOL:
            break

        # Newton step via vega, accepted only if it stays inside the bracket.
        vol_sqrt_t = sigma * math.sqrt(t_years)
        d1 = (
            math.log(spot / strike) + (rate - div_yield + 0.5 * sigma * sigma) * t_years
        ) / vol_sqrt_t
        vega = disc_spot * _norm_pdf(d1) * math.sqrt(t_years)
        if vega > 1e-12:
            step = f / vega
            candidate = sigma - step
            if lo < candidate < hi:
                sigma = candidate
                continue
        sigma = 0.5 * (lo + hi)

    return sigma


@dataclass(frozen=True)
class Contract:
    """One gathered option print, reduced to what the inversion needs.

    ``mid`` (the bid/ask midpoint) is the preferred mark when the gather stored
    one -- it is a cleaner input than the last print. Legacy POC cells
    (data/options-poc) carry no ``mid`` and keep inverting on ``close``.
    """

    strike: float
    close: float
    is_call: bool
    mid: float | None = None

    @property
    def price(self) -> float:
        """Effective price to invert: the MID when present, else the close."""
        return self.mid if self.mid is not None else self.close

    @classmethod
    def from_sample(cls, raw: dict) -> Contract:
        """Build from a samples.jsonl contract object.

        ``type`` is authoritative for call/put; we fall back to the ``role``
        naming (``otm_put`` -> put) if a print omits it. ``mid`` is read when the
        cell carries a non-null one (the ThetaData gather path); a cell without a
        ``mid`` key (the older POC path) leaves it None and inverts on ``close``.
        """
        kind = str(raw.get("type", "")).lower()
        role = str(raw.get("role", "")).lower()
        if kind in ("call", "put"):
            is_call = kind == "call"
        else:
            is_call = "put" not in role
        raw_mid = raw.get("mid")
        mid = float(raw_mid) if raw_mid is not None else None
        return cls(
            strike=float(raw["strike"]), close=float(raw["close"]), is_call=is_call, mid=mid
        )


@dataclass(frozen=True)
class SkewResult:
    """Per-cell inverted vols and the two skew measures.

    Any leg may be ``None`` when its price could not be inverted; a skew is
    ``None`` whenever either of its legs is ``None``.
    """

    iv_atm: float | None
    iv_otm_put: float | None
    iv_otm_call: float | None
    skew_put_atm: float | None
    skew_put_call: float | None


def contract_iv(
    contract: Contract,
    spot: float,
    days_to_expiry: float,
    rate: float = RATE,
    div_yield: float = DIV_YIELD,
) -> float | None:
    """Implied vol for a single gathered contract (from its MID when it has one,
    else its close -- see ``Contract.price``)."""
    t_years = days_to_expiry / 365.0
    return implied_vol(
        contract.price,
        spot,
        contract.strike,
        t_years,
        rate=rate,
        div_yield=div_yield,
        is_call=contract.is_call,
    )


def compute_skew(
    spot: float,
    days_to_expiry: float,
    atm: Contract,
    otm_put: Contract,
    otm_call: Contract | None = None,
    rate: float = RATE,
    div_yield: float = DIV_YIELD,
) -> SkewResult:
    """Invert the legs and form the skew measures.

    * ``skew_put_atm  = iv_otm_put - iv_atm``  -- how much richer the downside
      put vol is than at-the-money (the classic equity "smirk"). Needs only the
      ATM and OTM-put legs.
    * ``skew_put_call = iv_otm_put - iv_otm_call`` -- a symmetric risk-reversal:
      downside vol minus equidistant upside vol. Needs the OTM-call leg; when it
      is absent (a two-leg sample) this measure is None but skew_put_atm still
      computes.

    A steep (large positive) put skew is the market paying up for downside
    protection; the hypothesis under test is that it precedes LOWER forward
    stock returns.
    """
    iv_atm = contract_iv(atm, spot, days_to_expiry, rate, div_yield)
    iv_otm_put = contract_iv(otm_put, spot, days_to_expiry, rate, div_yield)
    iv_otm_call = (
        contract_iv(otm_call, spot, days_to_expiry, rate, div_yield)
        if otm_call is not None
        else None
    )

    skew_put_atm = iv_otm_put - iv_atm if (iv_otm_put is not None and iv_atm is not None) else None
    skew_put_call = (
        iv_otm_put - iv_otm_call if (iv_otm_put is not None and iv_otm_call is not None) else None
    )
    return SkewResult(
        iv_atm=iv_atm,
        iv_otm_put=iv_otm_put,
        iv_otm_call=iv_otm_call,
        skew_put_atm=skew_put_atm,
        skew_put_call=skew_put_call,
    )


def skew_from_cell(cell: dict) -> SkewResult | None:
    """Convenience: compute skew straight from a parsed samples.jsonl cell.

    Requires the ATM and OTM-put legs (for skew_put_atm, the primary signal);
    the OTM-call leg is OPTIONAL -- a two-leg sample yields skew_put_atm with
    skew_put_call None. Returns ``None`` only when atm or otm_put is missing.
    """
    # Drop legs the data source gap-filled: Robinhood's expired-option daily
    # history flags interpolated bars (often a $0.01 placeholder), which are not
    # real prices and would invert to garbage IV. A leg missing for this reason
    # is treated as absent. The threshold is applied to the EFFECTIVE price (the
    # mid when the cell carries one, else the close) so a leg the gather accepted
    # on a good mid is not dropped for a stale sub-threshold close, and vice-versa.
    by_role: dict[str, dict] = {}
    for raw in cell.get("contracts", []):
        raw_mid = raw.get("mid")
        effective = float(raw_mid) if raw_mid is not None else float(raw.get("close", 0.0))
        if raw.get("interpolated") or effective <= _MIN_OPTION_CLOSE:
            continue
        role = str(raw.get("role", "")).lower()
        by_role[role] = raw
    if not {"atm", "otm_put"} <= by_role.keys():
        return None
    atm_raw = by_role["atm"]
    otm_call = by_role.get("otm_call")
    # Collapsed ladder: when the OTM-call snapped onto the ATM strike the
    # risk-reversal degenerates into a clone of skew_put_atm. Drop the leg so
    # skew_put_call comes back None rather than a duplicate signal.
    if otm_call is not None and float(otm_call["strike"]) == float(atm_raw["strike"]):
        otm_call = None
    return compute_skew(
        spot=float(cell["spot_at_decision"]),
        days_to_expiry=float(cell["days_to_expiry"]),
        atm=Contract.from_sample(atm_raw),
        otm_put=Contract.from_sample(by_role["otm_put"]),
        otm_call=Contract.from_sample(otm_call) if otm_call is not None else None,
    )
