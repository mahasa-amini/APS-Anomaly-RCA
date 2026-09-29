"""Fixed Gaussian pair mechanisms, independent partitions, and pair ground truth."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Settings:
    n_fit: int = 1024
    n_control: int = 256
    n_evaluation: int = 1024
    n_pairs: int = 4
    healthy_rho: float = 0.8
    changed_rhos: tuple = (0.0, -0.8)
    missing_probability: float = 0.05


@dataclass
class Evaluation:
    observed: pd.DataFrame
    latent_before: np.ndarray
    latent_after: np.ndarray
    affected_pairs: np.ndarray


@dataclass
class Data:
    healthy_fit: pd.DataFrame
    healthy_control: pd.DataFrame
    evaluations: dict
    pairs: tuple


def gaussian_pairs(innovations, rho):
    """Each output coordinate is N(0,1); independent innovations for each pair."""
    out = innovations.copy()
    out[..., 1] = rho * innovations[..., 0] + np.sqrt(1 - rho ** 2) * innovations[..., 1]
    return out


def generate(seed, settings=Settings()):
    if min(settings.n_fit, settings.n_control, settings.n_evaluation) < 2 or settings.n_pairs < 4:
        raise ValueError("Require at least two rows per partition and four pairs.")
    if not 0 < settings.healthy_rho < 1 or not 0 <= settings.missing_probability <= 1:
        raise ValueError("Invalid healthy correlation or missingness probability.")
    if not settings.changed_rhos or len(set(settings.changed_rhos)) != len(settings.changed_rhos):
        raise ValueError("Require distinct change scenarios.")
    if any(not -1 < rho <= 0 for rho in settings.changed_rhos):
        raise ValueError("Change correlations must be nonsingular and nonpositive.")
    p = settings.n_pairs
    names = tuple(f"signal_{i}" for i in range(2 * p)) + ("constant", "all_missing")
    pairs = tuple((2 * j, 2 * j + 1) for j in range(p))
    means, scales = np.linspace(-2, 2, 2 * p), np.linspace(0.5, 2, 2 * p)
    streams = np.random.SeedSequence(seed).spawn(2 + len(settings.changed_rhos))

    def latent(rng, n):
        return gaussian_pairs(rng.normal(size=(n, p, 2)), settings.healthy_rho)

    def physical(z):
        n = len(z)
        return np.column_stack((z.reshape(n, 2 * p) * scales + means,
                                np.full(n, 1209600.0), np.zeros(n)))

    def observe(rng, x, label):
        out = x.copy()
        out[rng.random(out.shape) < settings.missing_probability] = np.nan
        out[:, -1] = np.nan
        return pd.DataFrame(out, columns=names, index=[f"{seed}:{label}:{i}" for i in range(len(x))])

    fit_rng, control_rng = [np.random.default_rng(s) for s in streams[:2]]
    fit = observe(fit_rng, physical(latent(fit_rng, settings.n_fit)), "fit")
    control = observe(control_rng, physical(latent(control_rng, settings.n_control)), "control")
    evaluations = {}
    for rho, stream in zip(settings.changed_rhos, streams[2:]):
        rng = np.random.default_rng(stream)
        before = latent(rng, settings.n_evaluation)
        targets = rng.integers(p, size=settings.n_evaluation)
        after = before.copy()
        # Replace both members with fresh draws; neither is an initiating sensor.
        after[np.arange(len(after)), targets] = gaussian_pairs(rng.normal(size=(len(after), 2)), rho)
        name = f"changed_rho_{rho:g}"
        before, after = physical(before), physical(after)
        evaluations[name] = Evaluation(observe(rng, after, name), before, after, targets)
    return Data(fit, control, evaluations, pairs)
