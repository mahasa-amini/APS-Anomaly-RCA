# Root Cause Analysis Engine for Multivariate Industrial Systems

## Abstract

This project proposes a multi-layer Root Cause Analysis (RCA) framework for highly imbalanced industrial anomaly detection scenarios such as the APS dataset.

Unlike traditional anomaly detection systems that provide only binary failure predictions, this framework decomposes failure states across three analytical layers:

1. Structural dependency disruption
2. Feature-level contribution attribution
3. Failure archetype discovery
4. Graph-level system topology shifts

The proposed architecture integrates generative modeling, discriminative modeling, explainability methods, unsupervised learning, and graph comparison into a unified and modular RCA engine.

---

## Problem Statement

Industrial systems generate high-dimensional multivariate sensor data.  
While anomaly detection models can classify failure events, they fail to answer critical interpretability questions:

- Which sensors structurally contributed to the failure?
- Did inter-sensor dependencies break?
- Are there multiple latent failure modes?
- Does the system topology change during failure states?

This project addresses the interpretability gap between anomaly detection and root cause reasoning.

---

# Methodology

The RCA engine is structured into three layers, each capturing failure dynamics at a different scale.

---

## Layer 1 — Dependency-Based Structural Analysis (Gaussian Copula)

We model the joint dependency structure of healthy system behavior using a Gaussian Copula.

**Key Idea:**  
Failures are not merely marginal feature deviations but structural dependency disruptions.

### Procedure

1. Fit Copula on healthy samples.
2. Estimate likelihood deviation for anomaly samples.
3. Decompose structural breakdown into feature-level contributions.
4. Rank features based on dependency disruption magnitude.

This layer captures multivariate structural shifts beyond independent feature deviations.

---

## Layer 2 — SHAP-Based Failure Archetype Discovery

Anomalies are not homogeneous. We cluster them based on explanation vectors.

### Procedure

1. Train classifier (XGBoost / RandomForest).
2. Compute SHAP explanations for anomaly samples.
3. Embed SHAP vectors via UMAP or PCA.
4. Perform clustering (KMeans / HDBSCAN).
5. Characterize clusters using top SHAP + Copula features.

This layer reveals latent failure archetypes within the anomaly class.

---

## Layer 3 — Graph-Level Dependency Shift Analysis

We compare dependency graphs between healthy and failure states.

### Procedure

1. Select relevant features (Copula + SHAP ranking).
2. Learn correlation-based dependency graph (healthy).
3. Learn graph (failure).
4. Detect edge-level structural shifts:
   - Emergent edges
   - Disappeared edges
   - Weight perturbations

This provides system-level causal interpretation.

---

# Synthetic Cross-Domain Validation

To demonstrate generalizability, a synthetic multivariate industrial time-series dataset was generated:

- 170 correlated sensors
- Sinusoidal base signals
- Stochastic noise
- Controlled failure injection:
  - Noise explosion
  - Drift
  - Dependency breakdown
  - Offset jump

Failure ratio ≈ 2% to match APS imbalance.

Time-series were transformed into tabular format via sliding window feature extraction.

This validates cross-domain robustness of the RCA engine.

---

# Contributions

1. Dependency-driven root cause modeling
2. Archetype discovery via explanation clustering
3. Multi-scale failure interpretation
4. Cross-domain generalization (tabular & time-series)
5. Imbalance-aware design
6. Modular research-oriented architecture

---

# Methodological Positioning

The framework integrates:

- Generative modeling (Gaussian Copula)
- Discriminative modeling (XGBoost / RandomForest)
- Explainability (SHAP)
- Manifold learning (UMAP / PCA)
- Unsupervised clustering
- Graph-based structural comparison

The layered design enables multi-resolution root cause interpretation.

---

# Tech Stack

- Python 3.10+
- NumPy / Pandas
- Scikit-learn
- XGBoost
- SHAP
- UMAP
- NetworkX
- Matplotlib

---

# Reproducibility

All experiments use a fixed random seed (42).  
The architecture is modular and reproducible under consistent environment settings.

---

# Limitations

- Correlation-based graph learning does not imply true causality.
- Gaussian Copula assumes elliptical dependency structure.
- Static analysis does not capture temporal causal dynamics.

---

# Future Research Directions

- Non-linear causal discovery (NOTEARS / LiNGAM)
- Temporal causal modeling
- Online adaptive RCA
- Information-theoretic dependency modeling
- Streaming anomaly interpretability

---

# Conclusion

This project elevates anomaly detection into a structured, interpretability-driven Root Cause Analysis framework.

It bridges:

- Statistical dependency modeling
- Model explanation
- Unsupervised structure discovery
- Graph-level reasoning

The architecture is suitable for industrial monitoring, cyber-physical systems, IoT platforms, and predictive maintenance research.

---

Author: Mahasa Amini  
Project: APS RCA Engine  
