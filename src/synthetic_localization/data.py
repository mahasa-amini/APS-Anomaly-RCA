"""Deterministic, disjoint synthetic partitions and explicit injection bookkeeping."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GeneratorSettings:
    n_fit: int = 512
    n_control: int = 256
    n_evaluation: int = 512
    n_signal: int = 8
    shift_sizes: tuple = (2.0, 4.0)
    signal_missing: float = 0.05
    constant_missing: float = 0.10
    nuisance_missing: float = 0.50


@dataclass
class Evaluation:
    observed: pd.DataFrame
    latent_before: np.ndarray
    latent_after: np.ndarray
    injected_indices: np.ndarray
    injected_deltas: np.ndarray


@dataclass
class SyntheticData:
    healthy_fit: pd.DataFrame
    healthy_control: pd.DataFrame
    evaluations: dict


def generate(seed, settings=GeneratorSettings()):
    if min(settings.n_fit, settings.n_control, settings.n_evaluation, settings.n_signal) < 2:
        raise ValueError("Partition sizes and signal feature count must be at least two.")
    if not settings.shift_sizes or len(set(settings.shift_sizes)) != len(settings.shift_sizes):
        raise ValueError("Shift sizes must be nonempty and unique.")
    if any(not np.isfinite(s) or s <= 0 for s in settings.shift_sizes):
        raise ValueError("Shift sizes must be positive and finite.")
    if any(not 0 <= p <= 1 for p in (settings.signal_missing, settings.constant_missing, settings.nuisance_missing)):
        raise ValueError("Missingness probabilities must be in [0, 1].")
    names = [f"signal_{j}" for j in range(settings.n_signal)] + ["constant", "missing_nuisance", "all_missing"]
    means = np.linspace(-2.0, 2.0, settings.n_signal)
    scales = np.linspace(0.5, 2.0, settings.n_signal)
    streams = np.random.SeedSequence(seed).spawn(2 + len(settings.shift_sizes))

    def latent(rng, n):
        return np.column_stack([rng.normal(size=(n, settings.n_signal)) * scales + means,
                                np.full(n, 1209600.0), rng.normal(size=n), np.zeros(n)])

    def observe(rng, values, partition):
        rates = np.array([settings.signal_missing] * settings.n_signal +
                         [settings.constant_missing, settings.nuisance_missing, 1.0])
        out = values.copy()
        out[rng.random(values.shape) < rates] = np.nan
        return pd.DataFrame(out, columns=names,
                            index=[f"{seed}:{partition}:{i}" for i in range(len(out))])

    fit_rng, control_rng = (np.random.default_rng(s) for s in streams[:2])
    fit = observe(fit_rng, latent(fit_rng, settings.n_fit), "fit")
    control = observe(control_rng, latent(control_rng, settings.n_control), "control")
    evaluations = {}
    for amplitude, stream in zip(settings.shift_sizes, streams[2:]):
        rng = np.random.default_rng(stream)
        before = latent(rng, settings.n_evaluation)
        indices = rng.integers(settings.n_signal, size=settings.n_evaluation)
        deltas = rng.choice([-1.0, 1.0], size=settings.n_evaluation) * amplitude * scales[indices]
        after = before.copy()
        after[np.arange(len(after)), indices] += deltas
        name = f"marginal_shift_{amplitude:g}sigma"
        evaluations[name] = Evaluation(observe(rng, after, name), before, after, indices, deltas)
    return SyntheticData(fit, control, evaluations)
