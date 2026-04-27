"""
F1 Race Strategy Simulator  (v3 - Multi-Driver / Any Race)
===========================================================
Uses FastF1 race data to model tyre degradation and simulate the
optimal pit-stop window for a chosen stint.

NEW in v3:
  - Configurable YEAR / TRACK / SESSION / DRIVERS list at the top
  - Session loaded once; each driver extracted independently
  - Auto-selects the longest valid stint per driver
  - Graceful handling of DNFs, missing data, and short stints
  - Per-driver base-case analysis + driver comparison overlay plot
  - Scenario sensitivity analysis kept intact from v2

Data flow:
    FastF1 session (loaded once)
        -> per-driver laps  -> clean  -> select best stint
        -> fit degradation  -> simulate_strategy (reused)
        -> find_crossover   -> print summary
        -> driver comparison plot  +  per-driver scenario dashboard
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import fastf1


# =============================================================================
# 1.  GLOBAL CONFIGURATION  — edit here to analyse any race / driver set
# =============================================================================

YEAR           = 2023
TRACK          = "Bahrain"
SESSION        = "R"

# List every driver abbreviation you want to compare.
# Leave as a single-item list to analyse just one driver.
DRIVERS        = ["VER", "HAM", "ALO"]

# Stint selection
# Set to None to auto-select the longest stint for each driver.
# Set to an integer (e.g. 2) to force a specific stint number.
STINT_NUMBER   = None
MIN_STINT_LAPS = 3     # skip stints shorter than this (too few points to fit)

# Compound filtering (fair comparison)
# Set to a compound string to restrict ALL drivers to that compound:
#   "SOFT" | "MEDIUM" | "HARD" | None (no restriction, pick longest stint freely)
# When set, only laps of this compound are included before model fitting.
TARGET_COMPOUND = "HARD"   # <- change to None to disable filtering

MAX_EXTENSIONS = 30    # simulate extending up to this many laps (30 for high-deg races)
OUTLIER_CUTOFF = 120   # discard laps slower than this (pit-in / safety-car laps)
                       # Bahrain race laps ~95-100s; Monza ~85s -- set 120 for safety

# Backward-compat aliases used inside plot_all_scenarios suptitle
SESSION_YEAR  = YEAR
SESSION_RACE  = TRACK
SESSION_TYPE  = SESSION


# =============================================================================
# 2.  SCENARIO DEFINITION
#     Each Scenario is a self-contained set of parameters.
#     Add or edit entries in SCENARIOS to explore new cases.
# =============================================================================

@dataclass
class Scenario:
    """All tunable parameters for a single strategy scenario."""
    name:                   str
    pit_loss_sec:           float        # fixed time lost in the pit lane (s)
    fresh_tyre_adv:         float        # lap-1 time gain on fresh rubber (s)
    degradation_multiplier: float        # scale the fitted slope (1.0 = real data)
    stint_number:           int          # which stint to use as the data source
    color_stay:             str = "#e8534a"   # plot colour for "stay out" line
    color_pit:              str = "#4ab8e8"   # plot colour for "pit now" line
    # filled automatically after data loading:
    base_slope:             float = field(default=0.0,  init=False, repr=False)
    effective_slope:        float = field(default=0.0,  init=False, repr=False)
    base_lap_time:          float = field(default=0.0,  init=False, repr=False)
    compound:               str   = field(default="",   init=False, repr=False)


# --- Define the four scenarios -----------------------------------------------
SCENARIOS: list[Scenario] = [
    Scenario(
        name                  = "1. Base Case (real data)",
        pit_loss_sec          = 22.0,
        fresh_tyre_adv        = 0.8,
        degradation_multiplier= 1.0,
        stint_number          = 2,
        color_stay            = "#e8534a",
        color_pit             = "#4ab8e8",
    ),
    Scenario(
        name                  = "2. High Degradation (x3 slope)",
        pit_loss_sec          = 22.0,
        fresh_tyre_adv        = 0.8,
        degradation_multiplier= 3.0,     # simulates soft tyres / hot track
        stint_number          = 2,
        color_stay            = "#ff9f43",
        color_pit             = "#48dbfb",
    ),
    Scenario(
        name                  = "3. Low Pit Loss (18s)",
        pit_loss_sec          = 18.0,    # faster pit crew / shorter pit lane
        fresh_tyre_adv        = 0.8,
        degradation_multiplier= 1.0,
        stint_number          = 2,
        color_stay            = "#ff6b81",
        color_pit             = "#7bed9f",
    ),
    Scenario(
        name                  = "4. Soft Tyre (x3 deg + 1.5s gain)",
        pit_loss_sec          = 22.0,
        fresh_tyre_adv        = 1.5,     # soft compound gives bigger initial boost
        degradation_multiplier= 3.0,
        stint_number          = 2,
        color_stay            = "#a29bfe",
        color_pit             = "#fd79a8",
    ),
]


# =============================================================================
# 3.  DATA LOADING & CLEANING
# =============================================================================

def load_session(year: int, track: str, session_type: str):
    """
    Load a FastF1 session object once and return it.
    The session is shared across all drivers to avoid redundant downloads.
    """
    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    print(f"  Loading {year} {track} {session_type} ...", end=" ", flush=True)
    sess = fastf1.get_session(year, track, session_type)
    sess.load()
    print("done.")
    return sess


def extract_driver_laps(session, driver: str) -> pd.DataFrame:
    """
    Extract cleaned lap data for a single driver from an already-loaded session.

    Returns a DataFrame with columns:
        LapNumber | Stint | Compound | LapTimeSec
    Returns an empty DataFrame if the driver has no valid quick laps.
    """
    try:
        laps = session.laps.pick_drivers([driver]).pick_quicklaps()
        if laps.empty:
            return pd.DataFrame()
        df = laps[["LapNumber", "Stint", "Compound", "LapTime"]].copy()
        df["LapTimeSec"] = df["LapTime"].dt.total_seconds()
        return df
    except Exception as exc:
        print(f"    [WARN] Could not extract laps for {driver}: {exc}")
        return pd.DataFrame()


def load_session_data(year: int, race: str, session_type: str,
                      driver: str) -> pd.DataFrame:
    """
    Legacy single-call loader (v1/v2 API).  Kept for backward compatibility.
    Internally calls load_session + extract_driver_laps.
    """
    sess = load_session(year, race, session_type)
    return extract_driver_laps(sess, driver)


def clean_stints(df: pd.DataFrame, outlier_cutoff: float) -> pd.DataFrame:
    """Remove outlier laps (pit-out laps, safety-car bunching, etc.)."""
    before = len(df)
    df = df[df["LapTimeSec"] < outlier_cutoff].copy()
    removed = before - len(df)
    if removed:
        print(f"    Cleaned {removed} outlier lap(s) (>{outlier_cutoff}s). "
              f"Remaining: {len(df)} laps.")
    return df


def select_best_stint(stints: pd.DataFrame,
                      preferred: Optional[int],
                      min_laps: int) -> Optional[int]:
    """
    Choose which stint number to analyse.

    Priority:
      1. If 'preferred' is set and has >= min_laps, use it.
      2. Otherwise auto-select the stint with the most laps.
      3. Return None if no valid stint exists.
    """
    if stints.empty:
        return None

    counts = stints.groupby("Stint")["LapNumber"].count()
    valid  = counts[counts >= min_laps]
    if valid.empty:
        return None

    if preferred is not None:
        if preferred in valid.index:
            return int(preferred)
        print(f"    [WARN] Stint {preferred} has <{min_laps} laps or doesn't exist. "
              f"Auto-selecting longest stint instead.")

    # pick the stint with the most laps
    best = int(valid.idxmax())
    return best


def filter_by_compound(
    stints: pd.DataFrame,
    target_compound: Optional[str],
) -> pd.DataFrame:
    """
    Filter the laps DataFrame to only include rows matching target_compound.

    If target_compound is None, returns the DataFrame unchanged (no filtering).
    If no laps match, returns an empty DataFrame — the caller must handle this.

    Compound matching is case-insensitive and strips whitespace.
    """
    if target_compound is None:
        return stints  # no restriction requested

    mask = stints["Compound"].str.strip().str.upper() == target_compound.strip().upper()
    filtered = stints[mask].copy()

    n_removed = len(stints) - len(filtered)
    if n_removed:
        print(f"    Compound filter ({target_compound.upper()}): "
              f"kept {len(filtered)} laps, dropped {n_removed} laps "
              f"from other compounds.")
    return filtered


# =============================================================================
# 4.  DEGRADATION MODELLING  (unchanged from v1)
# =============================================================================

def fit_degradation_model(stint_df: pd.DataFrame) -> tuple[float, float]:
    """
    Fit a linear model:  lap_time = slope * lap_number + intercept
    via numpy.polyfit (degree 1).

    Returns
    -------
    slope     : seconds of extra time lost per additional lap
    intercept : y-intercept of the fit line
    """
    x = stint_df["LapNumber"].values
    y = stint_df["LapTimeSec"].values
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


# =============================================================================
# 5.  STRATEGY SIMULATION  (unchanged from v1)
# =============================================================================

def simulate_strategy(
    base_lap_time:        float,
    degradation_slope:    float,
    fresh_tyre_advantage: float,
    pit_loss:             float,
    max_extensions:       int,
) -> pd.DataFrame:
    """
    Compare cumulative race time for staying out vs pitting, for each
    possible stint extension from 1 to max_extensions laps.

    STAY OUT:   lap_time(n) = base + slope * n  (degrading every lap)
    PIT NOW:    pit_loss  +  sum((base - fresh_adv) + slope * n)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                             fresh tyres are faster but also degrade

    Returns DataFrame: laps_extended | stay_out_total | pit_now_total
    """
    records = []
    for n in range(1, max_extensions + 1):
        stay_out_total = sum(
            base_lap_time + degradation_slope * lap
            for lap in range(1, n + 1)
        )
        fresh_base  = base_lap_time - fresh_tyre_advantage
        fresh_total = sum(
            fresh_base + degradation_slope * lap
            for lap in range(1, n + 1)
        )
        pit_now_total = pit_loss + fresh_total
        records.append({
            "laps_extended":  n,
            "stay_out_total": stay_out_total,
            "pit_now_total":  pit_now_total,
        })
    return pd.DataFrame(records)


def find_crossover(sim_df: pd.DataFrame) -> Optional[int]:
    """
    Return the first lap extension where pitting becomes cumulatively faster.
    Returns None if no crossover exists within the simulation window.
    """
    cross = sim_df[sim_df["pit_now_total"] < sim_df["stay_out_total"]]
    if cross.empty:
        return None
    return int(cross.iloc[0]["laps_extended"])


# =============================================================================
# 6.  SCENARIO RUNNER  (new in v2)
# =============================================================================

def run_scenario(scenario: Scenario, stints: pd.DataFrame,
                 max_extensions: int) -> tuple[pd.DataFrame, Optional[int]]:
    """
    Populate a Scenario's derived fields from the cleaned stints DataFrame,
    then call simulate_strategy() and find_crossover().

    Parameters
    ----------
    scenario       : Scenario dataclass (mutated in-place for derived fields)
    stints         : cleaned laps DataFrame for all stints
    max_extensions : number of lap extensions to simulate

    Returns
    -------
    sim_df         : DataFrame with simulation results
    crossover_lap  : int or None
    """
    stint_df = stints[stints["Stint"] == scenario.stint_number].copy()
    if stint_df.empty:
        available = sorted(stints["Stint"].unique())
        raise ValueError(
            f"Stint {scenario.stint_number} not found in data. "
            f"Available: {available}"
        )

    # Store metadata back on the scenario object
    scenario.compound      = stint_df["Compound"].mode().iloc[0]
    scenario.base_lap_time = float(stint_df["LapTimeSec"].iloc[0])

    base_slope, _ = fit_degradation_model(stint_df)
    scenario.base_slope      = base_slope
    scenario.effective_slope = base_slope * scenario.degradation_multiplier

    # Run the core simulation (reusing v1 function unchanged)
    sim_df = simulate_strategy(
        base_lap_time        = scenario.base_lap_time,
        degradation_slope    = scenario.effective_slope,
        fresh_tyre_advantage = scenario.fresh_tyre_adv,
        pit_loss             = scenario.pit_loss_sec,
        max_extensions       = max_extensions,
    )
    crossover_lap = find_crossover(sim_df)
    return sim_df, crossover_lap


# =============================================================================
# 7.  PRINTING  (v1 table kept; scenario summary added)
# =============================================================================

def print_results(sim_df: pd.DataFrame,
                  crossover_lap: Optional[int]) -> None:
    """Pretty-print the simulation table and the optimal pit window."""
    header = (f"{'Ext (laps)':>10}  {'Stay Out (s)':>13}  "
              f"{'Pit Now (s)':>11}  {'Delta(s)':>9}")
    print("\n" + "-" * len(header))
    print(header)
    print("-" * len(header))

    for _, row in sim_df.iterrows():
        delta = row["stay_out_total"] - row["pit_now_total"]
        flag  = " <-- PIT" if row["laps_extended"] == crossover_lap else ""
        print(
            f"{int(row['laps_extended']):>10}  "
            f"{row['stay_out_total']:>13.2f}  "
            f"{row['pit_now_total']:>11.2f}  "
            f"{delta:>+9.2f}"
            f"{flag}"
        )

    print("-" * len(header))

    if crossover_lap:
        print(f"\n[OK] Optimal pit window: extend by {crossover_lap} lap(s) "
              f"then box -- pitting is faster from here.\n")
    else:
        print("\n[!]  No crossover in this window -- staying out is optimal.\n")


def print_scenario_summary(scenario: Scenario,
                           crossover_lap: Optional[int]) -> None:
    """
    One-line diagnosis per scenario explaining the crossover result
    and the key reason (degradation vs pit-loss balance).
    """
    eff  = scenario.effective_slope
    loss = scenario.pit_loss_sec
    adv  = scenario.fresh_tyre_adv

    # Theoretical laps needed to recover pit loss purely from fresh-tyre advantage
    # (ignoring degradation difference for a quick sanity figure)
    if adv > 0:
        theoretical_laps = loss / adv
    else:
        theoretical_laps = float("inf")

    print(f"  Scenario : {scenario.name}")
    print(f"  Compound : {scenario.compound}  |  "
          f"Base slope: {scenario.base_slope:+.4f} s/lap  |  "
          f"Effective slope: {eff:+.4f} s/lap  (x{scenario.degradation_multiplier})")
    print(f"  Pit loss : {loss}s  |  Fresh-tyre gain: {adv}s/lap")
    print(f"  Approx laps to recover pit loss (naive): {theoretical_laps:.1f} laps")

    if crossover_lap:
        print(f"  RESULT   : Crossover at lap +{crossover_lap}  => "
              f"BOX on lap {crossover_lap}")
    else:
        laps_needed = loss / (adv + eff * MAX_EXTENSIONS / 2) if (adv + eff) > 0 else 999
        print(f"  RESULT   : No crossover within {MAX_EXTENSIONS} laps.  "
              f"Reason: degradation ({eff:.4f} s/lap) too low to "
              f"offset {loss}s pit loss within the window.")
    print()


# =============================================================================
# 8.  VISUALISATION  (v1 single-plot kept; 2x2 dashboard added)
# =============================================================================

def _draw_scenario_axis(ax, sim_df: pd.DataFrame,
                        crossover_lap: Optional[int],
                        scenario: Scenario) -> None:
    """
    Draw a single strategy comparison panel (one scenario).
    Reusable for the 2x2 dashboard.
    """
    x    = sim_df["laps_extended"]
    stay = sim_df["stay_out_total"]
    pit  = sim_df["pit_now_total"]

    ax.plot(x, stay, color=scenario.color_stay, linewidth=2.2,
            marker="o", markersize=4, label="Stay Out")
    ax.plot(x, pit,  color=scenario.color_pit,  linewidth=2.2,
            marker="s", markersize=4, label="Pit Now")

    if crossover_lap is not None:
        cross_y = sim_df.loc[sim_df["laps_extended"] == crossover_lap,
                             "pit_now_total"].iloc[0]
        ax.axvline(crossover_lap, color="#f5c518", linestyle="--",
                   linewidth=1.6, alpha=0.9, zorder=3)
        ax.scatter([crossover_lap], [cross_y], color="#f5c518",
                   s=120, zorder=5, label=f"Crossover: +{crossover_lap} laps")
        ax.axvspan(crossover_lap, x.max(), alpha=0.07, color="#f5c518", zorder=1)
        ax.annotate(
            f" Box +{crossover_lap}L",
            xy=(crossover_lap, cross_y),
            xytext=(crossover_lap + 0.6, cross_y + (stay.max() - stay.min()) * 0.06),
            fontsize=8, color="#f5c518",
            arrowprops=dict(arrowstyle="->", color="#f5c518", lw=1.2),
        )
    else:
        # No crossover: add text note
        mid_y = (stay.min() + stay.max()) / 2
        ax.text(x.max() * 0.55, mid_y,
                "No crossover\n(stay out)",
                color="#aaaaaa", fontsize=8,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="#1f1f1f", ec="#555555", alpha=0.7))

    # Title block
    eff_slope = scenario.effective_slope
    mult_tag  = (f"  [x{scenario.degradation_multiplier} deg]"
                 if scenario.degradation_multiplier != 1.0 else "")
    ax.set_title(
        f"{scenario.name}\n"
        f"slope={eff_slope:+.4f} s/lap{mult_tag}  |  "
        f"pit={scenario.pit_loss_sec}s  |  gain={scenario.fresh_tyre_adv}s",
        fontsize=9, color="white", pad=8,
    )
    ax.set_xlabel("Laps Extended", fontsize=9)
    ax.set_ylabel("Cumulative Time (s)", fontsize=9)
    ax.legend(fontsize=8, framealpha=0.25, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.18)
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#2e2e2e")


def plot_all_scenarios(results: list[tuple[Scenario,
                                           pd.DataFrame,
                                           Optional[int]]],
                       title: str = "") -> None:
    """
    Render a 2x2 subplot dashboard — one panel per scenario — and save to disk.
    Accepts an optional title string for the suptitle.
    """
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.patch.set_facecolor("#0d0d0d")
    axes = axes.flatten()

    for ax in axes:
        ax.set_facecolor("#141414")

    for i, (scenario, sim_df, crossover_lap) in enumerate(results):
        _draw_scenario_axis(axes[i], sim_df, crossover_lap, scenario)

    fig.suptitle(
        title or (
            f"F1 Strategy Sensitivity Analysis  |  "
            f"{SESSION_YEAR} {SESSION_RACE} GP  |  Stint {STINT_NUMBER}"
        ),
        fontsize=14, color="white", y=1.01,
    )
    plt.tight_layout(rect=[0, 0, 1, 1])

    out_path = os.path.join(os.path.dirname(__file__), "scenario_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Dashboard saved -> {out_path}")
    plt.show()


def plot_strategy(sim_df: pd.DataFrame,
                  crossover_lap: Optional[int],
                  stint_num: int,
                  compound: str,
                  slope: float,
                  pit_loss: float = 22.0,
                  fresh_adv: float = 0.8) -> None:
    """
    Single-scenario plot (v1 API preserved for backward compatibility).
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#141414")

    x    = sim_df["laps_extended"]
    stay = sim_df["stay_out_total"]
    pit  = sim_df["pit_now_total"]

    ax.plot(x, stay, color="#e8534a", linewidth=2.5, marker="o", markersize=5,
            label="Stay Out (worn tyres)")
    ax.plot(x, pit,  color="#4ab8e8", linewidth=2.5, marker="s", markersize=5,
            label=f"Pit Now (+{pit_loss}s loss, fresh tyres)")

    if crossover_lap is not None:
        cross_row = sim_df[sim_df["laps_extended"] == crossover_lap].iloc[0]
        cross_y   = cross_row["pit_now_total"]
        ax.axvline(crossover_lap, color="#f5c518", linestyle="--",
                   linewidth=1.8, alpha=0.85, zorder=3)
        ax.scatter([crossover_lap], [cross_y], color="#f5c518",
                   s=140, zorder=5, label=f"Optimal pit: +{crossover_lap} laps")
        ax.axvspan(crossover_lap, x.max(), alpha=0.07, color="#f5c518", zorder=1)
        ax.annotate(
            f"  Box! (+{crossover_lap} laps)",
            xy=(crossover_lap, cross_y),
            xytext=(crossover_lap + 0.5, cross_y + 1.5),
            fontsize=10, color="#f5c518",
            arrowprops=dict(arrowstyle="->", color="#f5c518", lw=1.4),
        )

    ax.set_xlabel("Laps Extended in Current Stint", fontsize=12, labelpad=8)
    ax.set_ylabel("Cumulative Time (seconds)",      fontsize=12, labelpad=8)
    ax.set_title(
        f"Race Strategy Simulator - Stint {stint_num} ({compound})\n"
        f"Degradation: {slope:+.4f} s/lap  |  Pit loss: {pit_loss}s  |  "
        f"Fresh-tyre advantage: {fresh_adv}s",
        fontsize=13, pad=14, color="white",
    )
    ax.legend(fontsize=11, framealpha=0.3, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.2)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333333")

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "strategy_plot.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Plot saved -> {out_path}")
    plt.show()


# =============================================================================
# 9.  MULTI-DRIVER PIPELINE  (new in v3)
# =============================================================================

def analyze_driver(
    driver: str,
    session,
    base_pit_loss: float = 22.0,
    base_fresh_adv: float = 0.8,
) -> Optional[dict]:
    """
    Run the full single-driver base-case analysis pipeline.

    Returns a result dict or None if the driver has insufficient data.
    Keys: driver, compound, stint_num, base_lap, slope, sim_df, crossover_lap
    """
    print(f"  Extracting laps for {driver} ...", end=" ", flush=True)
    raw = extract_driver_laps(session, driver)
    if raw.empty:
        print(f"no valid lap data found. Skipping.")
        return None
    print(f"{len(raw)} laps found.")

    stints = clean_stints(raw, OUTLIER_CUTOFF)
    if stints.empty:
        print(f"    [SKIP] {driver}: all laps removed as outliers.")
        return None

    # --- NEW: compound filtering (fair comparison) ---------------------------
    stints = filter_by_compound(stints, TARGET_COMPOUND)
    if stints.empty:
        label = TARGET_COMPOUND.upper() if TARGET_COMPOUND else "any"
        print(f"    [SKIP] {driver}: no laps on {label} compound after filtering.")
        return None
    # -------------------------------------------------------------------------

    stint_num = select_best_stint(stints, STINT_NUMBER, MIN_STINT_LAPS)
    if stint_num is None:
        print(f"    [SKIP] {driver}: no stint with >= {MIN_STINT_LAPS} laps.")
        return None

    stint_df = stints[stints["Stint"] == stint_num].copy()
    compound  = stint_df["Compound"].mode().iloc[0]
    base_lap  = float(stint_df["LapTimeSec"].iloc[0])

    slope, intercept = fit_degradation_model(stint_df)

    sim_df = simulate_strategy(
        base_lap_time        = base_lap,
        degradation_slope    = slope,
        fresh_tyre_advantage = base_fresh_adv,
        pit_loss             = base_pit_loss,
        max_extensions       = MAX_EXTENSIONS,
    )
    crossover_lap = find_crossover(sim_df)

    return {
        "driver":        driver,
        "compound":      compound,
        "stint_num":     stint_num,
        "n_laps":        len(stint_df),
        "base_lap":      base_lap,
        "slope":         slope,
        "intercept":     intercept,
        "sim_df":        sim_df,
        "crossover_lap": crossover_lap,
    }


def print_driver_summary(result: dict) -> None:
    """Print a formatted summary block for one driver's base-case result."""
    d  = result["driver"]
    cl = result["crossover_lap"]
    print(f"  Driver    : {d}")
    print(f"  Compound  : {result['compound']}   Stint {result['stint_num']}  "
          f"({result['n_laps']} laps)")
    print(f"  Base lap  : {result['base_lap']:.3f} s")
    print(f"  Deg slope : {result['slope']:+.4f} s/lap")
    if cl:
        print(f"  Strategy  : [OK] Pit after +{cl} laps -- crossover exists.")
    else:
        print(f"  Strategy  : [!]  No crossover in {MAX_EXTENSIONS} laps -- stay out.")
    print()


def plot_driver_comparison(
    driver_results: list[dict],
    track: str,
    year: int,
) -> None:
    """
    Overlay all drivers' 'stay out' vs 'pit now' cumulative times on one chart.
    Each driver gets its own colour pair; crossovers are marked per driver.
    """
    # Colour pairs (stay, pit) for up to 6 drivers
    PALETTES = [
        ("#e8534a", "#4ab8e8"),
        ("#f5c518", "#a29bfe"),
        ("#7bed9f", "#fd79a8"),
        ("#ff9f43", "#48dbfb"),
        ("#cd84f1", "#ffb8b8"),
        ("#67e480", "#e96900"),
    ]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#141414")

    for i, res in enumerate(driver_results):
        c_stay, c_pit = PALETTES[i % len(PALETTES)]
        drv = res["driver"]
        x    = res["sim_df"]["laps_extended"]
        stay = res["sim_df"]["stay_out_total"]
        pit  = res["sim_df"]["pit_now_total"]
        cl   = res["crossover_lap"]

        ax.plot(x, stay, color=c_stay, linewidth=2.2, linestyle="-",
                marker="o", markersize=3, label=f"{drv} Stay Out")
        ax.plot(x, pit,  color=c_pit,  linewidth=2.2, linestyle="--",
                marker="s", markersize=3, label=f"{drv} Pit Now")

        if cl is not None:
            cross_y = res["sim_df"].loc[
                res["sim_df"]["laps_extended"] == cl, "pit_now_total"
            ].iloc[0]
            ax.axvline(cl, color=c_stay, linestyle=":", linewidth=1.2, alpha=0.6)
            ax.scatter([cl], [cross_y], color=c_stay, s=100, zorder=5)
            ax.annotate(
                f"{drv} box+{cl}L",
                xy=(cl, cross_y),
                xytext=(cl + 0.3, cross_y - (stay.max() - stay.min()) * 0.05),
                fontsize=8, color=c_stay,
                arrowprops=dict(arrowstyle="->", color=c_stay, lw=1.0),
            )

    ax.set_xlabel("Laps Extended in Current Stint", fontsize=12, labelpad=8)
    ax.set_ylabel("Cumulative Time (seconds)", fontsize=12, labelpad=8)
    ax.set_title(
        f"Driver Strategy Comparison  |  {year} {track} GP\n"
        f"Base case: pit loss=22s | fresh-tyre gain=0.8s",
        fontsize=13, pad=12, color="white",
    )
    ax.legend(fontsize=9, framealpha=0.25, loc="upper left", ncol=2)
    ax.grid(True, linestyle="--", alpha=0.18)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333333")

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "driver_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Comparison plot saved -> {out_path}")
    plt.show()


# =============================================================================
# 10. MAIN ORCHESTRATION  (v3 — multi-driver)
# =============================================================================

def main() -> None:
    print("\n" + "=" * 60)
    print("   F1 Race Strategy Simulator  v3 - Multi-Driver")
    print("=" * 60 + "\n")

    cmp_label = f"compound filter = {TARGET_COMPOUND.upper()}" if TARGET_COMPOUND else "no compound filter"
    print(f"  Session : {YEAR} {TRACK} {SESSION}")
    print(f"  Drivers : {DRIVERS}")
    print(f"  Compare : {cmp_label}\n")

    # -- Step 1: load session once --------------------------------------------
    print("[ Step 1 ] Loading session (once for all drivers) ...")
    session = load_session(YEAR, TRACK, SESSION)

    # -- Step 2: per-driver base-case analysis --------------------------------
    print(f"\n[ Step 2 ] Analysing {len(DRIVERS)} driver(s): {DRIVERS}\n")
    driver_results = []

    for driver in DRIVERS:
        print("-" * 50)
        print(f"  DRIVER: {driver}")
        result = analyze_driver(driver, session)
        if result is None:
            continue
        print_driver_summary(result)
        print_results(result["sim_df"], result["crossover_lap"])
        driver_results.append(result)

    if not driver_results:
        print("[ERROR] No valid driver data found. Check DRIVERS list and session.")
        return

    # -- Step 3: driver comparison overview -----------------------------------
    # Warn if drivers ended up on different compounds (only when no filter set)
    compounds_used = {r["compound"] for r in driver_results}
    print("\n" + "=" * 60)
    if TARGET_COMPOUND:
        print(f"[ Step 3 ] Driver Comparison Summary  ({TARGET_COMPOUND.upper()} compound)\n")
    else:
        if len(compounds_used) > 1:
            print(f"[ Step 3 ] Driver Comparison Summary")
            print(f"  [WARN] Mixed compounds in results: {compounds_used}")
            print(f"         Set TARGET_COMPOUND to enforce a fair comparison.\n")
        else:
            print(f"[ Step 3 ] Driver Comparison Summary  (all on {compounds_used.pop()})\n")
    print(f"  {'Driver':<8}  {'Compound':<8}  {'Stint':<6}  "
          f"{'Base Lap':>10}  {'Slope':>12}  {'Crossover':>12}")
    print("  " + "-" * 64)
    for res in driver_results:
        cl = res["crossover_lap"]
        cross_str = f"+{cl} laps" if cl else "None (stay out)"
        print(f"  {res['driver']:<8}  {res['compound']:<8}  {res['stint_num']:<6}  "
              f"{res['base_lap']:>10.3f}s  "
              f"{res['slope']:>+10.4f} s/lap  "
              f"{cross_str:>12}")
    print()

    # -- Step 4: driver comparison overlay plot --------------------------------
    print("[ Step 4 ] Plotting driver comparison ...")
    plot_driver_comparison(driver_results, TRACK, YEAR)

    # -- Step 5: per-driver scenario sensitivity (first driver only) ----------
    print("\n[ Step 5 ] Scenario sensitivity analysis "
          f"(for {driver_results[0]['driver']}) ...")

    # Patch each scenario's stint_number to the auto-selected one
    chosen_stint = driver_results[0]["stint_num"]
    for sc in SCENARIOS:
        sc.stint_number = chosen_stint

    # Load that driver's cleaned stints for the scenario runner
    raw0   = extract_driver_laps(session, driver_results[0]["driver"])
    stints0 = clean_stints(raw0, OUTLIER_CUTOFF)

    scenario_results: list[tuple[Scenario, pd.DataFrame, Optional[int]]] = []
    for scenario in SCENARIOS:
        try:
            sim_df, crossover_lap = run_scenario(scenario, stints0, MAX_EXTENSIONS)
            scenario_results.append((scenario, sim_df, crossover_lap))
            print_scenario_summary(scenario, crossover_lap)
        except ValueError as e:
            print(f"    [SKIP] Scenario '{scenario.name}': {e}")

    if scenario_results:
        drv0 = driver_results[0]["driver"]
        title = (f"F1 Strategy Sensitivity  |  {YEAR} {TRACK} GP  |  "
                 f"{drv0}  |  Stint {chosen_stint}")
        print("[ Step 6 ] Plotting scenario dashboard ...")
        plot_all_scenarios(scenario_results, title=title)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
