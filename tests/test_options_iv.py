"""Unit tests for the Black-Scholes IV inversion and skew helpers.

All inputs are fabricated in-test; nothing touches the network or the gathered
data directory.
"""

from __future__ import annotations

import math

import pytest

from trading.research.options_iv import (
    DIV_YIELD,
    RATE,
    Contract,
    bs_price,
    compute_skew,
    implied_vol,
)


@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("sigma", [0.10, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("strike", [90.0, 100.0, 110.0])
def test_round_trip_recovers_sigma(is_call: bool, sigma: float, strike: float) -> None:
    """A price built at a known sigma inverts back to that sigma within 1e-3.

    Strikes are kept near the money: a deep-OTM leg at low vol has essentially
    no time value and is intentionally guarded out (see the reliability tests),
    so it is not a valid round-trip target.
    """
    spot, t = 100.0, 0.25
    price = bs_price(spot, strike, t, RATE, DIV_YIELD, sigma, is_call)
    recovered = implied_vol(price, spot, strike, t, RATE, DIV_YIELD, is_call)
    assert recovered is not None
    assert recovered == pytest.approx(sigma, abs=1e-3)


def test_put_call_parity() -> None:
    """BS call/put prices obey parity: C - P = S e^{-qt} - K e^{-rt}."""
    spot, strike, t, sigma = 100.0, 105.0, 0.5, 0.3
    call = bs_price(spot, strike, t, RATE, DIV_YIELD, sigma, is_call=True)
    put = bs_price(spot, strike, t, RATE, DIV_YIELD, sigma, is_call=False)
    lhs = call - put
    rhs = spot * math.exp(-DIV_YIELD * t) - strike * math.exp(-RATE * t)
    assert lhs == pytest.approx(rhs, abs=1e-9)


def test_none_below_intrinsic() -> None:
    """A call priced below its discounted intrinsic floor is not invertible."""
    spot, strike, t = 100.0, 80.0, 0.25
    intrinsic = spot * math.exp(-DIV_YIELD * t) - strike * math.exp(-RATE * t)
    assert implied_vol(intrinsic - 1.0, spot, strike, t, is_call=True) is None


def test_none_on_near_zero_price() -> None:
    """A near-$0 (deep-OTM, unreliable) print returns None rather than garbage."""
    spot, strike, t = 100.0, 200.0, 0.05
    assert implied_vol(1e-6, spot, strike, t, is_call=True) is None
    assert implied_vol(0.0, spot, strike, t, is_call=True) is None


def test_none_on_all_intrinsic_price() -> None:
    """A price equal to intrinsic (no time value) has no vol content -> None."""
    spot, strike, t = 100.0, 90.0, 0.25
    intrinsic = spot * math.exp(-DIV_YIELD * t) - strike * math.exp(-RATE * t)
    assert implied_vol(intrinsic, spot, strike, t, is_call=True) is None


def test_none_on_bad_inputs() -> None:
    assert implied_vol(float("nan"), 100.0, 100.0, 0.25) is None
    assert implied_vol(5.0, 100.0, 100.0, 0.0) is None
    assert implied_vol(5.0, -1.0, 100.0, 0.25) is None


def test_synthetic_skew_hand_checked() -> None:
    """Build three legs at known vols and confirm the skew arithmetic.

    ATM call @ 30% vol, OTM put @ 40% vol, OTM call @ 25% vol:
        skew_put_atm  = 0.40 - 0.30 = 0.10
        skew_put_call = 0.40 - 0.25 = 0.15
    """
    spot, dte = 100.0, 30.0
    t = dte / 365.0
    atm = Contract(
        strike=100.0, close=bs_price(spot, 100.0, t, RATE, DIV_YIELD, 0.30, True), is_call=True
    )
    otm_put = Contract(
        strike=90.0, close=bs_price(spot, 90.0, t, RATE, DIV_YIELD, 0.40, False), is_call=False
    )
    otm_call = Contract(
        strike=110.0, close=bs_price(spot, 110.0, t, RATE, DIV_YIELD, 0.25, True), is_call=True
    )

    result = compute_skew(spot, dte, atm, otm_put, otm_call)

    assert result.iv_atm == pytest.approx(0.30, abs=1e-3)
    assert result.iv_otm_put == pytest.approx(0.40, abs=1e-3)
    assert result.iv_otm_call == pytest.approx(0.25, abs=1e-3)
    assert result.skew_put_atm == pytest.approx(0.10, abs=2e-3)
    assert result.skew_put_call == pytest.approx(0.15, abs=2e-3)


def test_skew_none_when_a_leg_uninvertible() -> None:
    """If a leg's price is near-zero, that leg's IV and its skews are None."""
    spot, dte = 100.0, 30.0
    t = dte / 365.0
    atm = Contract(
        strike=100.0, close=bs_price(spot, 100.0, t, RATE, DIV_YIELD, 0.30, True), is_call=True
    )
    dead_put = Contract(strike=90.0, close=0.0, is_call=False)
    otm_call = Contract(
        strike=110.0, close=bs_price(spot, 110.0, t, RATE, DIV_YIELD, 0.25, True), is_call=True
    )

    result = compute_skew(spot, dte, atm, dead_put, otm_call)
    assert result.iv_otm_put is None
    assert result.skew_put_atm is None
    assert result.skew_put_call is None
    # The invertible ATM leg is still recovered.
    assert result.iv_atm == pytest.approx(0.30, abs=1e-3)


def test_contract_from_sample_infers_type() -> None:
    put = Contract.from_sample({"role": "otm_put", "strike": 90, "close": 1.2})
    call = Contract.from_sample({"role": "atm", "type": "call", "strike": 100, "close": 3.4})
    assert put.is_call is False
    assert call.is_call is True


def _poc_cell(contracts, spot=100.0, dte=30):
    return {
        "symbol": "X",
        "decision_date": "2024-01-02",
        "spot_at_decision": spot,
        "days_to_expiry": dte,
        "contracts": contracts,
    }


def test_skew_from_cell_two_leg_computes_put_atm_only():
    # A 2-leg sample (atm + otm_put, no otm_call) must still yield skew_put_atm;
    # skew_put_call is None without the call leg.
    from trading.research.options_iv import skew_from_cell

    s = skew_from_cell(
        _poc_cell(
            [
                {
                    "role": "atm",
                    "type": "call",
                    "strike": 100.0,
                    "close": 3.5,
                    "interpolated": False,
                },
                {
                    "role": "otm_put",
                    "type": "put",
                    "strike": 90.0,
                    "close": 1.2,
                    "interpolated": False,
                },
            ]
        )
    )
    assert s is not None
    assert s.skew_put_atm is not None
    assert s.skew_put_call is None


def test_skew_from_cell_drops_interpolated_leg():
    # An interpolated / placeholder otm_put leg is dropped -> atm+otm_put
    # requirement fails -> None (no garbage IV from a $0.01 gap-fill).
    from trading.research.options_iv import skew_from_cell

    assert (
        skew_from_cell(
            _poc_cell(
                [
                    {
                        "role": "atm",
                        "type": "call",
                        "strike": 100.0,
                        "close": 3.5,
                        "interpolated": False,
                    },
                    {
                        "role": "otm_put",
                        "type": "put",
                        "strike": 90.0,
                        "close": 0.01,
                        "interpolated": True,
                    },
                ]
            )
        )
        is None
    )


def test_skew_from_cell_interpolated_call_still_yields_put_atm():
    # A junk otm_call leg must not sink the whole cell: put_atm still computes.
    from trading.research.options_iv import skew_from_cell

    s = skew_from_cell(
        _poc_cell(
            [
                {
                    "role": "atm",
                    "type": "call",
                    "strike": 100.0,
                    "close": 3.5,
                    "interpolated": False,
                },
                {
                    "role": "otm_put",
                    "type": "put",
                    "strike": 90.0,
                    "close": 1.2,
                    "interpolated": False,
                },
                {
                    "role": "otm_call",
                    "type": "call",
                    "strike": 110.0,
                    "close": 0.01,
                    "interpolated": True,
                },
            ]
        )
    )
    assert s is not None
    assert s.skew_put_atm is not None
    assert s.skew_put_call is None
