#!/usr/bin/env python3
"""Fit the six timing strategies of the paper to every game record and select the AIC/BIC winner.

For each game record (one human, computer-use agent, or LLM playing ten levels against a
rule-based partner) every timing decision of the focal player is scored under six models
(Appendix B of the paper): Random, Constant Wait, Linear +/-Calibration (called "counting" in
the code), and Confidence-Bayesian +/-Calibration.  When the focal player played first, the
model scores the probability of playing inside the observed interval; when the partner played
first, it scores the probability that the focal player was still waiting.  The maximised log
likelihoods give AIC = 2K - 2 log L and BIC = K log n - 2 log L, compared only within a record.

Outputs (in --output-dir): raw_fits.csv, aic_winners.csv, bic_winners.csv,
aic_winner_proportions.csv, bic_winner_proportions.csv, exclusions.csv, run_metadata.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammainc, gammaincc

import analysis_common as common

HERE = Path(__file__).resolve().parent  # default paths are relative to this file
# Data layout of github.com/hulaboratory/the-mind: results/<model>_<prompt>/game_*.json and
# results/cua_raw_data/*.json.  Looked for next to this file, then at the repository root.
DATA_DIR = next((d / "results" for d in (HERE, HERE.parent) if (d / "results").is_dir()), HERE / "results")
TINY = 1e-300  # probability floor before taking logs

# Timing resolution delta of each data source: human/CUA logs are in milliseconds; LLM prompts
# advance in fixed steps (Section 3.3.3).
HUMAN_RESOLUTION_SECONDS = 0.001
PROMPT_RESOLUTIONS = {
    "time_based_q_second": 0.25,
    "time_based_half_second": 0.5,
    "time_based_second": 1.0,
    "time_based2": 0.001,
    "wait_based": 1.0,
    "wait_count_based": 1.0,
}

# Parameter grids and bounds (Appendix B.5, Equations 17-21).
TAU_GRID = (0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5)
ETA_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)
GAMMA_GRID = (0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 3.0)
KAPPA_BOUNDS = (0.02, 10000.0)
BETA0_BOUNDS = (-12.0, 6.0)
BETA1_BOUNDS = (0.0, 20.0)
QUADRATURE_NODES, QUADRATURE_WEIGHTS = np.polynomial.legendre.leggauss(8)

FAMILY_ORDER = [
    "random",
    "constant_wait",
    "counting_off",
    "counting_on",
    "confidence_bayesian_off",
    "confidence_bayesian_on",
]

FAMILY_LABELS = {
    "random": "Random",
    "constant_wait": "Constant wait",
    "counting_off": "Linear (calibration off)",
    "counting_on": "Linear (calibration on)",
    "confidence_bayesian_off": "Confidence Bayesian (calibration off)",
    "confidence_bayesian_on": "Confidence Bayesian (calibration on)",
}

# Number of fitted parameters K per model; grid-selected values count as fitted (Appendix B.5).
FAMILY_K = {
    "random": 1,  # p
    "constant_wait": 2,  # mu, kappa
    "counting_off": 2,  # tau, kappa
    "counting_on": 3,  # tau, eta, kappa
    "confidence_bayesian_off": 4,  # tau, gamma, beta0, beta1
    "confidence_bayesian_on": 5,  # tau, gamma, eta, beta0, beta1
}

# Computer-use agents: (label, default data file, command-line flag).
COMPUTER_USE_AGENTS = (
    ("Gemini", "the-mind-gemini_updated.json", "computer_use_gemini_data"),
    ("GPT-5.6 Luna", "the-mind-gpt-luna_updated.json", "computer_use_gpt_luna_data"),
    ("GPT-6", "the-mind-gpt6_updated.json", "computer_use_gpt6_data"),
)


# --------------------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------------------


@dataclass
class DecisionInterval:
    """One card play and the waiting interval that preceded it, seen from the focal player."""

    duration: float  # seconds waited since the previous card (or level start)
    resolution: float  # timing resolution delta of the data source
    focal_play: bool  # True if the focal player played, False if the partner did
    include_likelihood: bool  # False for forced autoplays, solo plays, invalid states
    own_lowest: int | None  # focal player's lowest remaining card a_1
    previous_card: int  # most recently played card c (0 at level start)
    prior_values: np.ndarray  # candidate values k of the partner's lowest card
    prior_probabilities: np.ndarray  # P(B_1 = k | history), Equation 5
    tempo_observation: float | None  # t_i / (c_i - c_{i-1}) used for calibration, Eq. 12


@dataclass
class GameRecord:
    target_id: str
    cohort: str  # "human", "computer_use", or "llm"
    llm_model: str
    prompt_type: str
    opponent_condition: str
    hand: int
    run_number: int
    source_file: str
    intervals: list[DecisionInterval]
    interval_exclusions: Counter[str]


@dataclass(frozen=True)
class LikelihoodArrays:
    """Array form of the included intervals, shared by all Gamma- and hazard-based fits."""

    durations: np.ndarray
    resolutions: np.ndarray
    focal_plays: np.ndarray


# --------------------------------------------------------------------------------------
# Game-state helpers
# --------------------------------------------------------------------------------------


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(HERE))
    except ValueError:
        return str(path)


def information_criterion(log_likelihood: float, k: int, n: int, criterion: str) -> float:
    """AIC = 2K - 2 log L, or BIC = K log n - 2 log L."""
    penalty = 2.0 * k if criterion == "aic" else math.log(n) * k
    return penalty - 2.0 * log_likelihood


def minimum_card_prior(
    own_remaining: Iterable[int],
    played_cards: Iterable[int],
    opponent_count: int,
    previous_card: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Prior over the partner's lowest remaining card (Equation 5).

    The partner's m cards are a uniform draw from the N cards that are unplayed, not in the focal
    hand, and above the last played card.  P(B_1 = k) = C(#cards above k, m - 1) / C(N, m).
    """
    own = {int(value) for value in own_remaining}
    played = {int(value) for value in played_cards}
    available = [
        value
        for value in range(1, 101)
        if value > previous_card and value not in own and value not in played
    ]
    n_available = len(available)
    if opponent_count <= 0 or n_available < opponent_count:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    log_denominator = (
        math.lgamma(n_available + 1)
        - math.lgamma(opponent_count + 1)
        - math.lgamma(n_available - opponent_count + 1)
    )
    values: list[float] = []
    log_probabilities: list[float] = []
    for rank, card in enumerate(available):
        greater = n_available - rank - 1
        if greater < opponent_count - 1:
            continue
        log_numerator = (
            math.lgamma(greater + 1)
            - math.lgamma(opponent_count)
            - math.lgamma(greater - opponent_count + 2)
        )
        values.append(float(card))
        log_probabilities.append(log_numerator - log_denominator)

    if not values:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    log_probs = np.asarray(log_probabilities, dtype=np.float64)
    log_probs -= float(np.max(log_probs))
    probabilities = np.exp(log_probs)
    probabilities /= float(probabilities.sum())
    return np.asarray(values, dtype=np.float64), probabilities


def likelihood_arrays(intervals: list[DecisionInterval]) -> LikelihoodArrays:
    included = [interval for interval in intervals if interval.include_likelihood]
    return LikelihoodArrays(
        durations=np.asarray([interval.duration for interval in included], dtype=np.float64),
        resolutions=np.asarray([interval.resolution for interval in included], dtype=np.float64),
        focal_plays=np.asarray([interval.focal_play for interval in included], dtype=np.bool_),
    )


def tempo_sequence(
    intervals: list[DecisionInterval], initial_tau: float, eta: float | None
) -> np.ndarray:
    """Tempo tau_i in force at each included interval.

    Without calibration tau stays fixed.  With calibration it is updated after every observed
    play by the exponential moving average tau_i = (1 - eta) tau_{i-1} + eta t_i/(c_i - c_{i-1})
    (Equation 12); the update is applied after the interval has been scored.
    """
    values: list[float] = []
    tau = initial_tau
    for interval in intervals:
        if interval.include_likelihood:
            values.append(tau)
        if eta is not None and interval.tempo_observation is not None:
            tau = (1.0 - eta) * tau + eta * interval.tempo_observation
    return np.asarray(values, dtype=np.float64)


# --------------------------------------------------------------------------------------
# Random model (Appendix B.3)
# --------------------------------------------------------------------------------------


def random_cdf(time_value: float, p: float) -> float:
    """CDF of T = N + U with N ~ Geometric(p) (zero based) and U ~ Uniform(0, 1), Equation 15."""
    if time_value <= 0.0:
        return 0.0
    whole = math.floor(time_value)
    fraction = time_value - whole
    survival_at_start = (1.0 - p) ** whole
    return 1.0 - survival_at_start + p * survival_at_start * fraction


def random_log_likelihood(data: LikelihoodArrays, p: float) -> float:
    total = 0.0
    for duration, resolution, focal in zip(data.durations, data.resolutions, data.focal_plays):
        if focal:
            probability = random_cdf(duration + resolution, p) - random_cdf(duration, p)
        else:
            probability = 1.0 - random_cdf(duration, p)
        total += math.log(max(probability, TINY))
    return total


def fit_random(data: LikelihoodArrays) -> dict[str, Any]:
    result = minimize_scalar(
        lambda p: -random_log_likelihood(data, float(p)),
        bounds=(1e-6, 1.0 - 1e-6),
        method="bounded",
        options={"xatol": 1e-10},
    )
    return {"log_likelihood": -float(result.fun), "parameters": {"p": float(result.x)}}


# --------------------------------------------------------------------------------------
# Gamma waiting-time models: Constant Wait (B.4) and Linear +/-Calibration (B.1)
# --------------------------------------------------------------------------------------


def gamma_log_likelihood_batch(
    data: LikelihoodArrays, predicted_times: np.ndarray, kappas: np.ndarray
) -> np.ndarray:
    """Log likelihood of T ~ Gamma(shape kappa, scale mean/kappa) for each row of predicted means.

    The mean is floored at delta/2 (Equation 11).  Focal plays score the probability mass in the
    observed bin [t, t + delta); partner plays score the survival probability P(T > t).
    """
    means = np.maximum(np.asarray(predicted_times, dtype=np.float64), data.resolutions[None, :] / 2.0)
    shapes = np.asarray(kappas, dtype=np.float64)[:, None]
    scales = means / shapes
    lower = gammainc(shapes, data.durations[None, :] / scales)
    upper = gammainc(shapes, (data.durations[None, :] + data.resolutions[None, :]) / scales)
    event_probabilities = np.maximum(upper - lower, 0.0)
    survival_probabilities = gammaincc(shapes, data.durations[None, :] / scales)
    probabilities = np.where(data.focal_plays[None, :], event_probabilities, survival_probabilities)
    return np.log(np.maximum(probabilities, TINY)).sum(axis=1)


def gamma_log_likelihood(data: LikelihoodArrays, predicted_times: np.ndarray, kappa: float) -> float:
    return float(gamma_log_likelihood_batch(data, predicted_times[None, :], np.asarray([kappa]))[0])


def fit_gamma_shapes_batch(
    data: LikelihoodArrays, predicted_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Golden-section search of the kappa MLE in log space, one search per row of predictions."""
    candidate_count = predicted_times.shape[0]
    lower = np.full(candidate_count, math.log(KAPPA_BOUNDS[0]), dtype=np.float64)
    upper = np.full(candidate_count, math.log(KAPPA_BOUNDS[1]), dtype=np.float64)
    inverse_phi = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - inverse_phi * (upper - lower)
    right = lower + inverse_phi * (upper - lower)
    left_values = gamma_log_likelihood_batch(data, predicted_times, np.exp(left))
    right_values = gamma_log_likelihood_batch(data, predicted_times, np.exp(right))
    for _ in range(32):
        choose_left = left_values > right_values
        new_lower = np.where(choose_left, lower, left)
        new_upper = np.where(choose_left, right, upper)
        new_point = np.where(
            choose_left,
            new_upper - inverse_phi * (new_upper - new_lower),
            new_lower + inverse_phi * (new_upper - new_lower),
        )
        new_values = gamma_log_likelihood_batch(data, predicted_times, np.exp(new_point))
        left, right, left_values, right_values = (
            np.where(choose_left, new_point, right),
            np.where(choose_left, left, new_point),
            np.where(choose_left, new_values, right_values),
            np.where(choose_left, left_values, new_values),
        )
        lower, upper = new_lower, new_upper
    kappas = np.exp((lower + upper) / 2.0)
    return gamma_log_likelihood_batch(data, predicted_times, kappas), kappas


def fit_constant_wait(data: LikelihoodArrays) -> dict[str, Any]:
    """Constant Wait: T_i ~ Gamma(kappa, mu/kappa) with mu and kappa fitted jointly (Equation 16)."""
    minimum_mu = max(float(np.max(data.resolutions)) / 2.0, 1e-6)
    maximum_mu = max(1000.0, float(np.max(data.durations + data.resolutions)) * 10.0)
    positive_times = data.durations[data.durations > 0]
    typical = float(np.median(positive_times)) if positive_times.size else minimum_mu
    starts = {
        (math.log(minimum_mu), math.log(0.5)),
        (math.log(max(typical, minimum_mu)), math.log(1.0)),
        (math.log(max(typical, minimum_mu)), math.log(4.0)),
    }
    best = None
    for start in starts:
        result = minimize(
            lambda values: -gamma_log_likelihood(
                data,
                np.full(len(data.durations), math.exp(float(values[0]))),
                math.exp(float(values[1])),
            ),
            np.asarray(start),
            method="L-BFGS-B",
            bounds=(
                (math.log(minimum_mu), math.log(maximum_mu)),
                (math.log(KAPPA_BOUNDS[0]), math.log(KAPPA_BOUNDS[1])),
            ),
        )
        if best is None or float(result.fun) < float(best.fun):
            best = result
    return {
        "log_likelihood": -float(best.fun),
        "parameters": {"mu": math.exp(float(best.x[0])), "kappa": math.exp(float(best.x[1]))},
    }


def linear_predicted_times(
    intervals: list[DecisionInterval], initial_tau: float, eta: float | None
) -> np.ndarray:
    """Linear model prediction tau_i * (a_1 - c) for every included interval (Equation 4)."""
    taus = tempo_sequence(intervals, initial_tau, eta)
    gaps = np.asarray(
        [
            max(interval.own_lowest - interval.previous_card, 0)
            for interval in intervals
            if interval.include_likelihood
        ],
        dtype=np.float64,
    )
    return taus * gaps


def fit_linear(
    record: GameRecord, data: LikelihoodArrays, calibrated: bool
) -> dict[str, Any]:
    """Grid over tau (and eta), continuous kappa MLE per grid point, keep the best candidate."""
    candidates = [
        {"tau": tau, "eta": eta}
        for tau in TAU_GRID
        for eta in (ETA_GRID if calibrated else (None,))
    ]
    predictions = np.stack(
        [linear_predicted_times(record.intervals, c["tau"], c["eta"]) for c in candidates]
    )
    likelihoods, kappas = fit_gamma_shapes_batch(data, predictions)
    best = int(np.argmax(likelihoods))
    return {
        "log_likelihood": float(likelihoods[best]),
        "parameters": {**candidates[best], "kappa": float(kappas[best])},
    }


# --------------------------------------------------------------------------------------
# Confidence-Bayesian hazard model (Appendix A.3 and B.2)
# --------------------------------------------------------------------------------------


def bayesian_confidence_values(
    interval: DecisionInterval, times: np.ndarray, tau: float, gamma: float
) -> np.ndarray:
    """P(B_1 > a_1 | partner silent until t) at each time t (Equations 6, 7, 9).

    Posterior weights are prior(k) * Weibull survival exp(-(t / (tau (k - c)))^gamma); the
    confidence is the posterior mass on partner cards above the focal player's lowest card.
    """
    gaps = interval.prior_values - float(interval.previous_card)
    log_weights = np.log(interval.prior_probabilities)[None, :] - np.power(
        times[:, None] / (tau * gaps[None, :]), gamma
    )
    log_weights -= np.max(log_weights, axis=1, keepdims=True)
    weights = np.exp(log_weights)
    numerator = weights[:, interval.prior_values > float(interval.own_lowest)].sum(axis=1)
    return numerator / weights.sum(axis=1)


def confidence_quadrature(
    intervals: list[DecisionInterval], taus: np.ndarray, gamma: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Confidence q(t) at Gauss-Legendre nodes over [0, t) and over the event bin [t, t + delta)."""
    included = [interval for interval in intervals if interval.include_likelihood]
    node_count = len(QUADRATURE_NODES)
    prefix_q = np.zeros((len(included), node_count), dtype=np.float64)
    bin_q = np.zeros((len(included), node_count), dtype=np.float64)
    prefix_scales = np.zeros(len(included), dtype=np.float64)
    bin_scales = np.zeros(len(included), dtype=np.float64)
    for index, (interval, tau) in enumerate(zip(included, taus)):
        prefix_scales[index] = interval.duration / 2.0
        bin_scales[index] = interval.resolution / 2.0
        if interval.duration > 0:
            prefix_times = prefix_scales[index] * (QUADRATURE_NODES + 1.0)
            prefix_q[index] = bayesian_confidence_values(interval, prefix_times, float(tau), gamma)
        bin_times = interval.duration + bin_scales[index] * (QUADRATURE_NODES + 1.0)
        bin_q[index] = bayesian_confidence_values(interval, bin_times, float(tau), gamma)
    return prefix_q, bin_q, prefix_scales, bin_scales


def confidence_hazard_log_likelihood(
    data: LikelihoodArrays,
    quadrature: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    beta0: float,
    beta1: float,
) -> float:
    """Hazard lambda(t) = exp(beta0 + beta1 q(t)); P(T > t) = exp(-integral of lambda).

    Partner plays score log P(T > t).  Focal plays score
    log [P(T > t) - P(T > t + delta)] = -Lambda(t) + log(1 - exp(-Lambda_bin)).
    """
    prefix_q, bin_q, prefix_scales, bin_scales = quadrature
    baseline = math.exp(beta0)
    cumulative_hazard = baseline * prefix_scales * (np.exp(beta1 * prefix_q) @ QUADRATURE_WEIGHTS)
    bin_hazard = baseline * bin_scales * (np.exp(beta1 * bin_q) @ QUADRATURE_WEIGHTS)
    event_bins = np.log(np.maximum(-np.expm1(-bin_hazard), TINY))
    log_probabilities = np.where(
        data.focal_plays, -cumulative_hazard + event_bins, -cumulative_hazard
    )
    return float(log_probabilities.sum())


def fit_confidence_hazard(
    data: LikelihoodArrays, quadrature: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> dict[str, Any]:
    """Fit beta0 and beta1 by L-BFGS-B from three starts (beta0 starts at the empirical rate)."""
    exposure = float(data.durations.sum() + np.sum(data.resolutions[data.focal_plays]) / 2.0)
    events = int(data.focal_plays.sum())
    initial_beta0 = math.log(max(events, 0.5) / max(exposure, 1e-6))
    initial_beta0 = min(max(initial_beta0, BETA0_BOUNDS[0]), BETA0_BOUNDS[1])
    best = None
    for initial_beta1 in (0.0, 2.0, 8.0):
        result = minimize(
            lambda values: -confidence_hazard_log_likelihood(
                data, quadrature, float(values[0]), float(values[1])
            ),
            np.asarray([initial_beta0, initial_beta1]),
            method="L-BFGS-B",
            bounds=(BETA0_BOUNDS, BETA1_BOUNDS),
        )
        if best is None or float(result.fun) < float(best.fun):
            best = result
    return {
        "log_likelihood": -float(best.fun),
        "parameters": {"beta0": float(best.x[0]), "beta1": float(best.x[1])},
    }


def fit_confidence_bayesian(
    record: GameRecord, data: LikelihoodArrays, calibrated: bool
) -> dict[str, Any]:
    """Grid over tau, gamma (and eta); continuous beta0, beta1 per grid point; keep the best."""
    best: dict[str, Any] | None = None
    for tau in TAU_GRID:
        for gamma in GAMMA_GRID:
            for eta in ETA_GRID if calibrated else (None,):
                taus = tempo_sequence(record.intervals, tau, eta)
                quadrature = confidence_quadrature(record.intervals, taus, gamma)
                fit = fit_confidence_hazard(data, quadrature)
                if best is None or fit["log_likelihood"] > best["log_likelihood"] + 1e-12:
                    best = {
                        "log_likelihood": fit["log_likelihood"],
                        "parameters": {"tau": tau, "gamma": gamma, "eta": eta, **fit["parameters"]},
                    }
    return best


# --------------------------------------------------------------------------------------
# Reconstructing timing decisions from game logs
# --------------------------------------------------------------------------------------


def reconstruct_initial_hand(actions: list[dict[str, Any]], focal_key: str, human: bool) -> set[int]:
    """Focal player's dealt hand for one level: every card they played plus cards seen in snapshots."""
    cards = {
        int(action["value"])
        for action in actions
        if (action.get("playerId") == "human" if human else action.get("player_id") == focal_key)
        and finite_number(action.get("value"))
    }
    for action in actions:
        snapshot = action.get("handsSnapshot") if human else action.get("left_cards")
        if not isinstance(snapshot, dict):
            continue
        if human:
            for key, values in snapshot.items():
                if str(key).lower() in {"partner", "agent"} or not isinstance(values, list):
                    continue
                cards.update(int(value) for value in values)
        else:
            values = snapshot.get(focal_key)
            if isinstance(values, list):
                cards.update(int(value) for value in values)
    return cards


def build_intervals(
    actions: list[dict[str, Any]], focal_key: str, human: bool, resolution: float
) -> tuple[list[DecisionInterval], Counter[str]]:
    """Replay one game record level by level and emit a DecisionInterval per card played.

    Human/CUA logs ("human" schema) and LLM logs differ in field names; both are handled here.
    Forced autoplays after a partner error and solo plays (partner has no cards) are kept for
    tempo calibration bookkeeping but excluded from the likelihood (Section 4.2).
    """
    intervals: list[DecisionInterval] = []
    exclusions: Counter[str] = Counter()
    levels = sorted({int(a["level"]) for a in actions if finite_number(a.get("level"))})
    for level in levels:
        level_actions = [a for a in actions if int(a.get("level", -1)) == level]
        own_remaining = reconstruct_initial_hand(level_actions, focal_key, human)
        if len(own_remaining) < level:
            exclusions["incomplete focal hand reconstruction"] += len(level_actions)
            continue
        opponent_count = level
        played: set[int] = set()
        for action in level_actions:
            actor_is_focal = (
                action.get("playerId") == "human" if human else action.get("player_id") == focal_key
            )
            forced = bool(action.get("autoPlayed")) if human else action.get("isError") == "AutoPlay"
            duration_raw = action.get("hesitationTime")
            if finite_number(duration_raw):
                duration = float(duration_raw) / 1000.0 if human else float(duration_raw)
            else:
                duration = math.nan
            previous_raw = action.get("maxBefore") if human else action.get("topBefore")
            previous_card = int(previous_raw) if finite_number(previous_raw) else max(played, default=0)
            own_lowest = min(own_remaining) if own_remaining else None
            solo = opponent_count <= 0 or (not human and bool(action.get("solo_play")))

            reason = ""
            if forced:
                reason = "forced autoplay"
            elif solo:
                reason = "solo play"
            elif own_lowest is None:
                reason = "focal hand empty"
            elif not math.isfinite(duration) or duration < 0:
                reason = "invalid waiting time"
            elif actor_is_focal and int(action.get("value", -1)) != own_lowest:
                reason = "focal action is not reconstructed lowest card"
            prior_values, prior_probabilities = minimum_card_prior(
                own_remaining, played, opponent_count, previous_card
            )
            if not reason and prior_values.size == 0:
                reason = "invalid Bayesian state"
            if reason:
                exclusions[reason] += 1

            gap = int(action.get("value", 0)) - previous_card
            tempo_observation = (
                duration / gap
                if not forced and not solo and math.isfinite(duration) and duration >= 0 and gap > 0
                else None
            )
            intervals.append(
                DecisionInterval(
                    duration=max(duration, 0.0) if math.isfinite(duration) else 0.0,
                    resolution=resolution,
                    focal_play=actor_is_focal,
                    include_likelihood=not reason,
                    own_lowest=own_lowest,
                    previous_card=previous_card,
                    prior_values=prior_values,
                    prior_probabilities=prior_probabilities,
                    tempo_observation=tempo_observation,
                )
            )

            value = action.get("value")
            if finite_number(value):
                card = int(value)
                played.add(card)
                if actor_is_focal:
                    own_remaining.discard(card)
                else:
                    opponent_count = max(0, opponent_count - 1)
    return intervals, exclusions


# --------------------------------------------------------------------------------------
# Loading the three cohorts
# --------------------------------------------------------------------------------------


def human_condition(record: dict[str, Any]) -> str:
    """Map the human-schema partner settings to one of the five partner conditions."""
    partner = str(record.get("partnerType", "")).lower()
    calibrated = bool(record.get("calibration"))
    if partner == "random":
        return "random"
    if partner == "rule":
        return "counting_on" if calibrated else "counting_off"
    if partner == "bayesian":
        return "bayesian_on" if calibrated else "bayesian_off"
    raise ValueError(f"Unrecognized human partner type: {partner}")


def load_human_schema_records(
    path: Path, cohort: str, label: str, excluded_ids: set[str]
) -> tuple[list[GameRecord], list[dict[str, Any]]]:
    """Load human or computer-use game records (both use the web-interface log schema)."""
    records: list[GameRecord] = []
    exclusions: list[dict[str, Any]] = []
    if not path.is_file():
        return records, [{"cohort": cohort, "target_id": label, "reason": f"missing data file: {path}"}]
    for record_id, record in json.loads(path.read_text()).items():
        target_id = f"{label}::{record_id}" if label else record_id
        if any(record_id.endswith(suffix) for suffix in excluded_ids):
            exclusions.append(
                {"cohort": cohort, "target_id": target_id, "reason": "pre-specified participant exclusion"}
            )
            continue
        try:
            intervals, interval_exclusions = build_intervals(
                list(record.get("roundData", [])), "human", True, HUMAN_RESOLUTION_SECONDS
            )
            records.append(
                GameRecord(
                    target_id=target_id,
                    cohort=cohort,
                    llm_model=label,
                    prompt_type="",
                    opponent_condition=human_condition(record),
                    hand=int(record.get("handId", 1)),
                    run_number=0,
                    source_file=relative_path(path),
                    intervals=intervals,
                    interval_exclusions=interval_exclusions,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            exclusions.append(
                {"cohort": cohort, "target_id": target_id, "reason": f"could not reconstruct event history: {exc}"}
            )
    return records, exclusions


def load_llm_records(
    results_dir: Path, prompt_type: str
) -> tuple[list[GameRecord], list[dict[str, Any]]]:
    """Load one prompt condition for every LLM, keeping the latest run per model/partner/hand."""
    candidates: list[GameRecord] = []
    exclusions: list[dict[str, Any]] = []
    resolution = PROMPT_RESOLUTIONS[prompt_type]
    for model_label, directory_prefix, separator in common.LLM_MODELS:
        directory = results_dir / f"{directory_prefix}{separator}{prompt_type}"
        if not directory.is_dir():
            exclusions.append({"cohort": "llm", "target_id": model_label, "reason": f"missing directory: {directory}"})
            continue
        for path in sorted(directory.glob("game_*.json")):
            relative = str(path.relative_to(results_dir))
            try:
                payload = json.loads(path.read_text())
                models = payload.get("models", {})
                strategy_keys = [k for k, v in models.items() if str(v).startswith(common.STRATEGY_PREFIXES)]
                focal_keys = [k for k in models if k not in strategy_keys]
                if len(strategy_keys) != 1 or len(focal_keys) != 1:
                    exclusions.append({"cohort": "llm", "target_id": relative, "reason": "self-play or ambiguous roles"})
                    continue
                intervals, interval_exclusions = build_intervals(
                    list(payload.get("action", {}).values()), focal_keys[0], False, resolution
                )
                candidates.append(
                    GameRecord(
                        target_id=relative,
                        cohort="llm",
                        llm_model=model_label,
                        prompt_type=prompt_type,
                        opponent_condition=common.condition_from_model(str(models[strategy_keys[0]])),
                        hand=common.hand_from_result_filename(path),
                        run_number=int(path.stem.rsplit("_", 1)[-1]),
                        source_file=relative_path(path),
                        intervals=intervals,
                        interval_exclusions=interval_exclusions,
                    )
                )
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                exclusions.append({"cohort": "llm", "target_id": relative, "reason": f"could not reconstruct event history: {exc}"})

    # Keep the highest-numbered run when a model/partner/hand cell was recorded more than once.
    grouped: dict[tuple[str, str, int], list[GameRecord]] = defaultdict(list)
    for record in candidates:
        grouped[(record.llm_model, record.opponent_condition, record.hand)].append(record)
    records: list[GameRecord] = []
    for key in sorted(grouped):
        selected = max(grouped[key], key=lambda r: (r.run_number, r.target_id))
        records.append(selected)
        for discarded in grouped[key]:
            if discarded is not selected:
                exclusions.append(
                    {
                        "cohort": "llm",
                        "target_id": discarded.target_id,
                        "reason": f"duplicate model/condition/hand; selected run {selected.run_number}: {selected.target_id}",
                    }
                )
    for model_label, _, _ in common.LLM_MODELS:
        for condition in common.CONDITION_ORDER:
            for hand in range(1, 11):
                if (model_label, condition, hand) not in grouped:
                    exclusions.append(
                        {
                            "cohort": "llm",
                            "target_id": f"{model_label}/{condition}/hand_{hand}",
                            "reason": "missing LLM hand in selected prompt",
                        }
                    )
    return records, exclusions



# --------------------------------------------------------------------------------------
# Fitting one record and summarising
# --------------------------------------------------------------------------------------


def fit_game(record: GameRecord) -> list[dict[str, Any]]:
    """Fit all six models to one game record; return one row per model with AIC/BIC and winners."""
    data = likelihood_arrays(record.intervals)
    n = len(data.durations)
    if n == 0:
        return []
    fits = {
        "random": fit_random(data),
        "constant_wait": fit_constant_wait(data),
        "counting_off": fit_linear(record, data, calibrated=False),
        "counting_on": fit_linear(record, data, calibrated=True),
        "confidence_bayesian_off": fit_confidence_bayesian(record, data, calibrated=False),
        "confidence_bayesian_on": fit_confidence_bayesian(record, data, calibrated=True),
    }
    scores = {
        criterion: [
            information_criterion(fits[f]["log_likelihood"], FAMILY_K[f], n, criterion)
            for f in FAMILY_ORDER
        ]
        for criterion in ("aic", "bic")
    }
    weights = {c: common.akaike_weights(scores[c]) for c in scores}
    # Ties go to the simpler model (earlier in FAMILY_ORDER).
    winners = {c: FAMILY_ORDER[min(range(len(FAMILY_ORDER)), key=lambda i: (scores[c][i], i))] for c in scores}
    play_count = int(data.focal_plays.sum())

    rows: list[dict[str, Any]] = []
    for index, family in enumerate(FAMILY_ORDER):
        rows.append(
            {
                "target_id": record.target_id,
                "cohort": record.cohort,
                "llm_model": record.llm_model,
                "prompt_type": record.prompt_type,
                "opponent_condition": record.opponent_condition,
                "hand": record.hand,
                "run_number": record.run_number,
                "source_file": record.source_file,
                "n_observations": n,
                "play_events": play_count,
                "censored_intervals": n - play_count,
                "excluded_intervals": sum(record.interval_exclusions.values()),
                "strategy_family": family,
                "strategy_label": FAMILY_LABELS[family],
                "best_parameters": fits[family]["parameters"],
                "k": FAMILY_K[family],
                "log_likelihood": fits[family]["log_likelihood"],
                "aic": scores["aic"][index],
                "bic": scores["bic"][index],
                "delta_aic": weights["aic"][0][index],
                "akaike_weight": weights["aic"][1][index],
                "delta_bic": weights["bic"][0][index],
                "bic_weight": weights["bic"][1][index],
                "aic_winner_family": winners["aic"],
                "bic_winner_family": winners["bic"],
                "is_aic_winner": family == winners["aic"],
                "is_bic_winner": family == winners["bic"],
            }
        )
    return rows


def winner_rows(rows: list[dict[str, Any]], criterion: str) -> list[dict[str, Any]]:
    """One row per game record: the model with the lowest AIC or BIC."""
    winners = [row for row in rows if row[f"is_{criterion}_winner"]]
    return sorted(
        winners,
        key=lambda row: (
            row["cohort"],
            row["llm_model"],
            row["prompt_type"],
            common.CONDITION_ORDER.index(row["opponent_condition"]),
            int(row["hand"]),
            row["target_id"],
        ),
    )


def winner_proportions(winners: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Share of game records won by each model within cohort / model / prompt / partner condition."""
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in winners:
        grouped[(row["cohort"], row["llm_model"], row["prompt_type"], row["opponent_condition"])].append(row)
    summary: list[dict[str, Any]] = []
    for key in sorted(grouped):
        counts = Counter(row["strategy_family"] for row in grouped[key])
        total = len(grouped[key])
        for family in FAMILY_ORDER:
            summary.append(
                {
                    "cohort": key[0],
                    "llm_model": key[1],
                    "prompt_type": key[2],
                    "opponent_condition": key[3],
                    "target_count": total,
                    "strategy_family": family,
                    "strategy_label": FAMILY_LABELS[family],
                    "winner_count": counts[family],
                    "winner_proportion": counts[family] / total,
                }
            )
    return summary


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--human-data", type=Path, default=DATA_DIR / "cua_raw_data" / "the-mind-human-agent.json")
    for _, filename, flag in COMPUTER_USE_AGENTS:
        parser.add_argument(f"--{flag.replace('_', '-')}", type=Path, default=DATA_DIR / "cua_raw_data" / filename)
    parser.add_argument("--results-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=HERE / "aic_generative_results_compact")
    parser.add_argument("--targets", choices=("all", "human", "llm", "computer_use"), default="all")
    parser.add_argument(
        "--llm-prompt-type",
        choices=tuple(prompt for prompt, _ in common.PROMPTS),
        default="time_based_second",
        help="LLM prompt condition to analyse (the paper reports the 1-second time-based prompt).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Game records fitted in parallel (default: up to 4).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records: list[GameRecord] = []
    exclusions: list[dict[str, Any]] = []
    if args.targets in ("all", "human"):
        loaded, excluded = load_human_schema_records(args.human_data, "human", "", common.HUMAN_EXCLUSIONS)
        records += loaded
        exclusions += excluded
        print(f"Loaded {len(loaded):,} human game records.", flush=True)
    if args.targets in ("all", "llm"):
        loaded, excluded = load_llm_records(args.results_dir, args.llm_prompt_type)
        records += loaded
        exclusions += excluded
        print(f"Loaded {len(loaded):,} LLM game records.", flush=True)
    if args.targets in ("all", "computer_use"):
        for label, _, flag in COMPUTER_USE_AGENTS:
            loaded, excluded = load_human_schema_records(getattr(args, flag), "computer_use", label, set())
            records += loaded
            exclusions += excluded
        print(f"Loaded {sum(r.cohort == 'computer_use' for r in records):,} computer-use game records.", flush=True)

    rows: list[dict[str, Any]] = []
    if args.jobs == 1:
        fitted = map(fit_game, records)
    else:
        executor = ProcessPoolExecutor(max_workers=args.jobs)
        fitted = executor.map(fit_game, records, chunksize=1)
    for index, (record, record_rows) in enumerate(zip(records, fitted), start=1):
        if not record_rows:
            exclusions.append({"cohort": record.cohort, "target_id": record.target_id, "reason": "zero eligible decision intervals"})
        rows += record_rows
        print(f"  fitted {index:,}/{len(records):,}: {record.cohort} {record.target_id}", flush=True)
    if args.jobs > 1:
        executor.shutdown()

    aic_winners = winner_rows(rows, "aic")
    bic_winners = winner_rows(rows, "bic")
    common.write_csv(args.output_dir / "raw_fits.csv", rows)
    common.write_csv(args.output_dir / "aic_winners.csv", aic_winners)
    common.write_csv(args.output_dir / "bic_winners.csv", bic_winners)
    common.write_csv(args.output_dir / "aic_winner_proportions.csv", winner_proportions(aic_winners))
    common.write_csv(args.output_dir / "bic_winner_proportions.csv", winner_proportions(bic_winners))
    common.write_csv(args.output_dir / "exclusions.csv", exclusions)

    interval_exclusions: Counter[str] = Counter()
    for record in records:
        interval_exclusions.update(record.interval_exclusions)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "settings": {
            "targets": args.targets,
            "llm_prompt_type": args.llm_prompt_type,
            "human_resolution_seconds": HUMAN_RESOLUTION_SECONDS,
            "llm_resolution_seconds": PROMPT_RESOLUTIONS[args.llm_prompt_type],
            "parameter_grids": {
                "tau": TAU_GRID,
                "eta": ETA_GRID,
                "gamma": GAMMA_GRID,
                "kappa_bounds": KAPPA_BOUNDS,
                "beta0_bounds": BETA0_BOUNDS,
                "beta1_bounds": BETA1_BOUNDS,
                "gauss_legendre_nodes": len(QUADRATURE_NODES),
            },
            "parameter_counts": FAMILY_K,
        },
        "counts": {
            "loaded_records_by_cohort": dict(Counter(r.cohort for r in records)),
            "fitted_records": len(aic_winners),
            "aic_bic_agreement": sum(r["aic_winner_family"] == r["bic_winner_family"] for r in aic_winners) / max(len(aic_winners), 1),
            "record_exclusions": dict(Counter(str(e["reason"]).split(";")[0] for e in exclusions)),
            "interval_exclusions": dict(interval_exclusions),
        },
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Complete: {len(aic_winners):,} fitted records in {time.monotonic() - started:.1f}s.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
