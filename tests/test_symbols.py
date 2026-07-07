"""Symbol resolution: rename chains, namespace overrides, and the
contamination guard (a membership old ticker must never resolve to its own
literal, recycled form)."""

import trading.symbols as symbols
from trading.symbols import NAMESPACE_OVERRIDES, RENAMES, normalize, resolve_current


def test_passthrough_for_unknown_symbol():
    assert resolve_current("AAPL") == "AAPL"
    assert resolve_current("NOSUCHTICKER") == "NOSUCHTICKER"


def test_normalize_convention():
    assert normalize(" brk.b ") == "BRK-B"
    assert resolve_current("brk.b") == "BRK-B"  # normalized even when not in a chain


def test_single_rename(monkeypatch):
    monkeypatch.setattr(symbols, "RENAMES", [("OLD", "NEW", "2020-01-01")])
    monkeypatch.setattr(symbols, "NAMESPACE_OVERRIDES", {})
    assert resolve_current("OLD") == "NEW"
    assert resolve_current("NEW") == "NEW"  # terminal is a no-op


def test_multi_hop_chain(monkeypatch):
    monkeypatch.setattr(
        symbols,
        "RENAMES",
        [("A", "B", "2019-01-01"), ("B", "C", "2021-01-01")],
    )
    monkeypatch.setattr(symbols, "NAMESPACE_OVERRIDES", {})
    assert resolve_current("A") == "C"  # A->B->C
    assert resolve_current("B") == "C"
    assert resolve_current("C") == "C"


def test_cycle_guard_terminates(monkeypatch):
    # A->B->A must not loop forever; the seen-set guard stops deterministically.
    monkeypatch.setattr(
        symbols,
        "RENAMES",
        [("A", "B", "2020-01-01"), ("B", "A", "2021-01-01")],
    )
    monkeypatch.setattr(symbols, "NAMESPACE_OVERRIDES", {})
    assert resolve_current("A") in {"A", "B"}  # terminates, doesn't hang
    assert resolve_current("B") in {"A", "B"}


def test_namespace_override_applied_after_chain(monkeypatch):
    monkeypatch.setattr(symbols, "RENAMES", [])
    monkeypatch.setattr(symbols, "NAMESPACE_OVERRIDES", {"MMC": "MRSH"})
    assert resolve_current("MMC") == "MRSH"
    assert resolve_current("mmc") == "MRSH"  # normalized first


def test_mmc_namespace_override_is_live():
    # The real committed override: Tiingo's bare MMC is the ASX Mitre Mining
    # squatter; US Marsh & McLennan is served under MRSH.
    assert NAMESPACE_OVERRIDES["MMC"] == "MRSH"
    assert resolve_current("MMC") == "MRSH"


def test_every_rename_old_ticker_redirects_away_from_its_literal():
    # Contamination guard: fetching the LITERAL old ticker risks a recycled,
    # different-identity company's bars. resolve_current MUST redirect every
    # RENAMES old ticker to something other than itself.
    for old, _new, _date in RENAMES:
        resolved = resolve_current(old)
        assert resolved != normalize(old), f"{old} did not redirect away from its literal form"


def test_rename_chains_resolve_to_a_terminal_not_in_the_old_set():
    # The terminal of every chain is a current ticker (never itself an old
    # ticker), so resolution reaches a fixed point in one call.
    olds = {normalize(o) for o, _, _ in RENAMES}
    for old, _new, _date in RENAMES:
        terminal = resolve_current(old)
        # After namespace override the terminal may be remapped; strip that.
        assert terminal not in olds or terminal in NAMESPACE_OVERRIDES.values()


def test_cbs_resolves_to_para_not_psky():
    # Chain deliberately terminates at PARA: PSKY is a fresh 2025-08-06 Tiingo
    # listing with no historical bars, so a CBS->PARA->PSKY hop would zero out
    # coverage. PARA carries the continuous 2006..2025 lineage.
    assert resolve_current("CBS") == "PARA"


def test_known_renames_reach_expected_successors():
    expected = {
        "ABC": "COR",
        "CTL": "LUMN",
        "ADS": "BFH",
        "BLL": "BALL",
        "COG": "CTRA",
        "GPS": "GAP",
        "HFC": "DINO",
        "TMK": "GL",
        "FB": "META",
    }
    for old, new in expected.items():
        assert resolve_current(old) == new
