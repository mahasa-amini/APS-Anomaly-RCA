"""Sample-wise injection recovery and eligibility-matched random references."""
import numpy as np


def rank_features(scores, eligible):
    scores, eligible = np.asarray(scores), np.asarray(eligible, dtype=bool)
    if scores.ndim != 2 or scores.shape != eligible.shape or not np.isfinite(scores).all():
        raise ValueError("Expected finite score and eligibility matrices of the same shape.")
    # Stable descending order resolves ties by original feature position.
    order = np.argsort(-np.where(eligible, scores, -np.inf), axis=1, kind="stable")
    return np.where(np.take_along_axis(eligible, order, axis=1), order, -1)


def localization(ranking, targets, k=3):
    ranking, targets = np.asarray(ranking), np.asarray(targets)
    if ranking.ndim != 2 or targets.shape != (len(ranking),) or len(targets) == 0 or k < 1:
        raise ValueError("Expected nonempty sample rankings, one injected index per sample, and k >= 1.")
    if not np.issubdtype(targets.dtype, np.integer) or (targets < 0).any() or (targets >= ranking.shape[1]).any():
        raise ValueError("Invalid injected feature indices.")
    hit1 = ranking[:, 0] == targets
    hitk = (ranking[:, :k] == targets[:, None]).any(axis=1)
    return dict(samples=len(targets), precision_at_1=float(hit1.mean()), recall_at_k=float(hitk.mean()),
                k=k, misses_at_1=int((~hit1).sum()), misses_at_k=int((~hitk).sum()))


def random_reference(eligible, targets, seed, k=3):
    rng = np.random.default_rng(seed)
    ranking = rank_features(rng.random(eligible.shape), eligible)
    measured = localization(ranking, targets, k)
    candidates = eligible.sum(axis=1)
    recoverable = eligible[np.arange(len(targets)), targets]
    denom = np.maximum(candidates, 1)
    measured["expected_precision_at_1"] = float((recoverable / denom).mean())
    measured["expected_recall_at_k"] = float((recoverable * np.minimum(k, candidates) / denom).mean())
    return measured
