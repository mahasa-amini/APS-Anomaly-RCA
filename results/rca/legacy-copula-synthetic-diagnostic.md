# RCA milestone 2: synthetic legacy Copula scoring diagnostic

The unchanged `GaussianCopulaDependencyAnalyzer` was fitted only on healthy toy
data. No APS CSV was read, no legacy pickle was loaded, and no saved pipeline
was executed. APS
context comes solely from the committed feature-quality summary. This milestone
diagnoses two separate issues; it does not change the RCA algorithms.

## Reproduce

From the repository root, recompute in memory and compare with the existing
JSON. This command does not write outputs and works with the committed JSON
already present; it fits only the synthetic toy analyzer in memory:

```sh
.venv/bin/python -B - <<'PY'
import json
from pathlib import Path
from scripts.diagnose_legacy_copula import diagnostic

saved = Path("results/rca/legacy-copula-synthetic-diagnostic.json")
result = diagnostic()
assert result == json.loads(saved.read_text()), "Recomputed diagnostic differs from saved JSON"
print("In-memory diagnostic matches the existing JSON; no output files written.")
print(json.dumps(result["constant_feature"], indent=2))
print(json.dumps(result["identity"], indent=2))
PY
```

Initial generation only, when the output JSON does **not** exist (do not run
this command to reproduce an already recorded diagnostic):

```sh
.venv/bin/python -B -m scripts.diagnose_legacy_copula
```

Run the four synthetic tests separately:

```sh
.venv/bin/python -B -m unittest discover -s tests -p test_legacy_copula_diagnostic.py -v
```

The initial-generation command creates
[the measured JSON](legacy-copula-synthetic-diagnostic.json) exclusively and
refuses to overwrite it. The in-memory equality check includes dependency
versions and hashes of the diagnostic, unchanged analyzer and prior audit;
use the recorded environment and unchanged source files.
No model or large intermediate is saved. Observations in the diagnostic are
synthetic, not APS records.

## 1. Unchanged constant feature receives a large attribution

With seed 42, generate 256 independent pairs of standard-normal draws and append
a column fixed at `1209600.0`. Fit the legacy analyzer with its default
regularization `epsilon=0.001`. Score three toy rows: one unchanged healthy row
and two rows whose varying coordinates are shifted by +2 and -2. The constant
coordinate remains exactly `1209600.0` in every row.

Measured with NumPy 2.5.3, pandas 3.0.6, SciPy 1.18.1 and scikit-learn 1.9.1:

| Quantity | Actual value |
|---|---:|
| Constant transformed value, all three rows | -5.199337582605575 |
| Healthy transformed constant mean | -5.199337582605575 |
| Healthy transformed constant variance | 0.0 |
| Regularized constant covariance diagonal | 0.001 |
| Constant precision diagonal | 1000.0 |
| Constant attribution, all three rows | 13516.555648947393 |

Inspection of the installed `QuantileTransformer._transform_col` explains the
boundary behavior: the constant is both the fitted minimum and maximum; the
lower-bound assignment occurs last. The normal inverse CDF is then clipped to
its finite lower bound. This produces a nonzero constant transformed value,
not a standard-normal marginal. Scikit-learn documents the normal-output
clipping in its [preprocessing guide](https://sklearn.org/stable/modules/preprocessing.html).

`np.cov` centers its inputs, so this constant contributes zero empirical
variance/cross-covariance. The regularizer supplies variance 0.001, but the
legacy log-density assumes mean zero rather than the observed transformed mean.
Replacing the coordinate by zero therefore removes a penalty of
`0.5 * (-5.199337582605575)^2 / 0.001 = 13516.555648947393`.
Even the unchanged healthy row receives that attribution. A synthetic regression
check also verifies that multiplying epsilon by ten divides this attribution
by ten, while the transformed value is unchanged.

This reproduces the proposed constant-feature numerical mechanism. It is close
to the earlier saved `f_89` mean (13516.555648947418), but does not authenticate
the old preprocessing history or prove every historical score's provenance.

## 2. Gaussian density is not dependence-only copula density

The current implementation computes, for transformed coordinates `z` and
regularized empirical covariance `Sigma`:

```text
ell_legacy(z) = -d/2 log(2*pi) - 1/2 log|Sigma| - 1/2 z^T Sigma^-1 z.
```

This is a zero-mean multivariate Gaussian log-density in **z-space**. It includes
marginal extremeness as well as dependence. It is also not a density in the
original feature coordinates: transformation Jacobians are not included.
Its attribution is coordinate replacement, not integration over a removed
feature: `Delta_j = ell(z with z_j=0) - ell(z)`. With `P=Sigma^-1`, expansion gives
`Delta_j = 0.5*P_jj*z_j^2 + z_j*sum_(k!=j)(P_jk*z_k)`.

For continuous margins, set `u_j=F_j(x_j)` and `z_j=Phi^-1(u_j)`. A Gaussian copula
uses a positive-definite **correlation** matrix `R`, with density relative to
uniform coordinates `u`:

```text
c_R(u) = phi_R(z) / product_j phi(z_j)
log c_R(u) = -1/2 log|R| - 1/2 z^T (R^-1 - I) z.
```

The division removes standard-normal marginal density terms. This is the density
ratio used in the [statsmodels copula source](https://www.statsmodels.org/stable/_modules/statsmodels/distributions/copula/elliptical.html).
The legacy `Sigma = empirical covariance + epsilon*I` is not necessarily
unit-diagonal, so it is not automatically a valid `R`. Simply subtracting
standard-normal marginal terms from that unnormalized covariance formula does
not establish a Gaussian copula density. The reference formula in this diagnostic
requires a symmetric, positive-definite, unit-diagonal matrix; it is not a patch
to the legacy analyzer.

Independently of fitting and the quantile transformer, set the analyzer's
covariance and precision to `I_2` and its log-determinant to zero:

| Formula | At z=(3,0) | At z=(0,0) | Replacement score |
|---|---:|---:|---:|
| Legacy Gaussian log-density | -6.337877066409345 | -1.8378770664093453 | 4.5 |
| Gaussian copula log-density with R=I | 0 | 0 | 0 |

Thus **4.5 = 3^2/2** is a purely marginal contribution despite zero modeled
dependence. This issue persists without any constant feature. In contrast,
the first issue is a degenerate marginal, boundary mapping and zero-mean density
interacting with a regularized near-zero variance. They must not be conflated.
Neither a Gaussian anomaly score nor a copula dependence score establishes a
physical cause: coordinate replacement is not a validated causal intervention,
and these observational models contain no identified physical causal mechanism.

## 3. APS modeling limitations from the existing aggregate audit

For the saved 48,000-row training split, using an explicit descriptive cutoff of
at most ten distinct observed values, the prior summary reports:

- `cd_000`: one observed value, 545 missing values.
- `ch_000`: two observed values, 11,870 missing values (24.729167%).
- 169 of 170 features have missing values; `br_000` has 39,438 missing values
  (82.1625%), the largest fraction.

Tied/discrete values do not become genuinely continuous standard-normal margins
through empirical quantile mapping. Imputation can add point masses and alter
estimated dependence; missingness may carry separate information. Continuous
copula-density formulas therefore need modeling justification for such inputs.
These are statistical limitations, not interpretations of physical sensor
meaning or mechanisms. No raw APS data was reopened for this context.

## Checks and scope

Four synthetic tests passed: unchanged-constant reproduction, inverse-epsilon
scaling, the independent identity-matrix example, and a nonidentity correlation
case checked against SciPy joint-minus-marginal log-densities. The legacy source,
completed Phase 1 artifacts/results and prior RCA outputs remain unchanged.
