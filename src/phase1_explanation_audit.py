"""Read-only, verified explanations of the recorded Phase 1 validation classifier."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import xgboost as xgb

from .data_loading import load_raw_aps
from .preprocessing import prepare_features

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = 'aps-develop-20260927T135520Z-a6474c'
MANIFEST_SHA256 = 'c50c6bad348cd6eace95cec8bc1508a53f9780cb0a8e3557cb8c48f9b4f1aa74'
ARTIFACT_NAMES = ('feature_names.json', 'split_indices.npz', 'imputer.joblib',
                  'model.ubj', 'validation_predictions.csv')
PROBABILITY_ATOL = 1e-7
MARGIN_ATOL = 2e-5
MARGIN_RTOL = 1e-6
TOP_K = 10


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_hash(path, expected):
    require(sha256(path) == expected, f'Hash mismatch: {Path(path).name}')


def input_paths(root):
    """Closed allowlist; never follow paths supplied by a saved manifest."""
    artifact = Path(root) / 'artifacts/phase1' / RUN_ID
    return dict(training=Path(root) / 'data/raw/aps_failure_training_set.csv',
                development=Path(root) / 'results/phase1' / RUN_ID / 'development.json',
                manifest=artifact / 'run_manifest.json',
                **{name: artifact / name for name in ARTIFACT_NAMES})


def verify_feature_order(actual, expected):
    require(list(actual) == list(expected), 'Feature names/order mismatch.')


def verify_split(train, validation, y, split):
    n = len(y)
    for name, indices in [('train', train), ('validation', validation)]:
        require(indices.ndim == 1 and np.issubdtype(indices.dtype, np.integer), f'Invalid {name} index type.')
        require(len(indices) > 0 and (indices >= 0).all() and (indices < n).all(), f'Invalid {name} index bounds.')
    combined = np.concatenate([train, validation])
    require(len(combined) == n and np.array_equal(np.sort(combined), np.arange(n)), 'Split indices overlap, omit or duplicate rows.')
    require(split['stratified'] is True and split['index_definition'] == 'zero-based data row, excluding CSV header', 'Unsupported split definition.')
    expected_train, expected_validation = train_test_split(np.arange(n), test_size=split['validation_size'],
                                                          stratify=y, random_state=split['seed'])
    require(np.array_equal(train, expected_train) and np.array_equal(validation, expected_validation),
            'Saved split order does not match the recorded deterministic split.')


def verify_predictions(frame, indices, labels, probability, threshold):
    require(list(frame.columns) == ['row_index', 'true', 'probability', 'predicted', 'predicted_0_5'],
            'Unexpected validation prediction columns.')
    require(np.array_equal(frame.row_index.to_numpy(), indices), 'Validation prediction indices/order mismatch.')
    require(np.array_equal(frame.true.to_numpy(), labels), 'Validation prediction labels mismatch.')
    saved = frame.probability.to_numpy(dtype=float)
    require(np.isfinite(saved).all() and ((saved >= 0) & (saved <= 1)).all(), 'Invalid stored probabilities.')
    error = float(np.max(np.abs(saved - probability)))
    require(np.allclose(saved, probability, atol=PROBABILITY_ATOL, rtol=0), 'Stored validation probability mismatch.')
    # Both stored and freshly reproduced probabilities must preserve fixed group membership.
    for cutoff, column in [(threshold, 'predicted'), (0.5, 'predicted_0_5')]:
        require(np.array_equal(frame[column].to_numpy(), (saved >= cutoff).astype(int)), 'Stored prediction rule mismatch.')
        require(np.array_equal(saved >= cutoff, probability >= cutoff), 'Reproduced prediction crosses a recorded threshold.')
    return saved, error


def verify_additivity(contributions, margins, saved_probability):
    require(contributions.ndim == 2 and margins.shape == (len(contributions),), 'Unexpected contribution/margin shape.')
    require(np.isfinite(contributions).all() and np.isfinite(margins).all(), 'Nonfinite model outputs.')
    reconstructed = contributions.astype(np.float64).sum(axis=1)
    require(np.allclose(reconstructed, margins, atol=MARGIN_ATOL, rtol=MARGIN_RTOL),
            'Contributions plus bias do not reconstruct the raw margin.')
    converted = expit(margins.astype(np.float64))
    require(np.allclose(converted, saved_probability, atol=PROBABILITY_ATOL, rtol=0),
            'Sigmoid(raw margin) does not reproduce the stored probability.')
    return dict(samples=len(margins), maximum_margin_absolute_error=float(np.max(np.abs(reconstructed-margins))),
                maximum_sigmoid_probability_absolute_error=float(np.max(np.abs(converted-saved_probability))))


def native_contributions(booster, matrix, features, saved_probability):
    """Exact native TreeSHAP, all saved trees, validation matrix only; no background fit."""
    if booster.feature_names is not None:
        verify_feature_order(booster.feature_names, features)
    dmatrix = xgb.DMatrix(matrix, feature_names=features, missing=np.nan, nthread=1)
    margins = booster.predict(dmatrix, output_margin=True, strict_shape=True, iteration_range=(0, 0))
    require(margins.shape == (len(matrix), 1), 'Expected binary single-margin output.')
    require(np.allclose(expit(margins[:, 0].astype(float)), saved_probability, atol=PROBABILITY_ATOL, rtol=0),
            'Native margin probability mismatch before explanations.')
    contributions = booster.predict(dmatrix, pred_contribs=True, approx_contribs=False,
                                    strict_shape=True, iteration_range=(0, 0))
    require(contributions.shape == (len(matrix), 1, len(features)+1), 'Unexpected native contribution output shape.')
    verification = verify_additivity(contributions[:, 0, :], margins[:, 0], saved_probability)
    return contributions[:, 0, :], verification


def feature_summary(values, raw, feature, index, rank=None):
    n = len(values)
    missing = int(raw.isna().sum())
    record = dict(feature=feature, feature_index=index, samples=n, raw_missing_count=missing,
                  raw_missing_fraction=missing/n if n else None,
                  mean_absolute_contribution=float(np.abs(values).mean()) if n else None,
                  mean_signed_contribution=float(values.mean()) if n else None,
                  mean_positive_part=float(np.maximum(values, 0).mean()) if n else None,
                  mean_negative_part=float(np.minimum(values, 0).mean()) if n else None,
                  positive_count=int((values > 0).sum()), negative_count=int((values < 0).sum()),
                  zero_count=int((values == 0).sum()))
    if rank is not None:
        record['rank_by_mean_absolute'] = rank
    return record


def summarize(X_validation, matrix, contributions, labels, probability, threshold):
    values = contributions[:, :-1].astype(float)
    predicted = probability >= threshold
    masks = dict(TP=(labels == 1) & predicted, FP=(labels == 0) & predicted,
                 FN=(labels == 1) & ~predicted, TN=(labels == 0) & ~predicted)
    groups = {}
    for name, mask in masks.items():
        n = int(mask.sum())
        order = np.argsort(-np.abs(values[mask]).mean(axis=0), kind='stable') if n else np.array([], dtype=int)
        groups[name] = dict(samples=n, bias=dict(samples=n, mean=float(contributions[mask, -1].astype(float).mean()) if n else None),
            top_features=[feature_summary(values[mask, j], X_validation.iloc[mask, j], X_validation.columns[j], int(j), rank)
                          for rank, j in enumerate(order[:TOP_K], 1)])
    require('cd_000' in X_validation.columns, 'cd_000 missing from verified schema.')
    j = list(X_validation.columns).index('cd_000')
    raw = X_validation.iloc[:, j]
    observed = raw.dropna()
    constant = dict(samples=len(raw), observed_count=len(observed), observed_unique=int(observed.nunique()),
                    observed_minimum=float(observed.min()) if len(observed) else None,
                    observed_maximum=float(observed.max()) if len(observed) else None,
                    imputed_unique=int(np.unique(matrix[:, j]).size),
                    imputed_minimum=float(matrix[:, j].min()), imputed_maximum=float(matrix[:, j].max()),
                    maximum_absolute_contribution=float(np.abs(values[:, j]).max()),
                    summary=feature_summary(values[:, j], raw, 'cd_000', j),
                    by_group={name: feature_summary(values[mask, j], raw.iloc[mask], 'cd_000', j) for name, mask in masks.items()})
    return groups, constant


def audit(root=ROOT, *, expected_manifest_sha256=MANIFEST_SHA256, expected_features=170):
    """Read only the approved inputs; return row-free output, never write files."""
    paths = input_paths(root)
    for path in paths.values():
        require(path.is_file(), f'Required input missing: {path.name}')
        require(path.resolve() == path.absolute(), f'Symlinked input refused: {path.name}')
    initial_hashes = {name: sha256(path) for name, path in paths.items()}
    require(initial_hashes['manifest'] == expected_manifest_sha256, 'Development manifest anchor hash mismatch.')
    development, manifest = read_json(paths['development']), read_json(paths['manifest'])
    require(development['run_id'] == manifest['run_id'] == RUN_ID, 'Run ID mismatch.')
    require(development['stage'] == 'development' and manifest['status'] == 'development_complete', 'Incomplete development run.')
    require(development['manifest_sha256'] == expected_manifest_sha256, 'Development summary manifest hash mismatch.')
    for key in ('data', 'split', 'model_params', 'imputation'):
        require(development[key] == manifest[key], f'Development/manifest {key} mismatch.')
    for name in ARTIFACT_NAMES:
        require(name in manifest['files'] and initial_hashes[name] == manifest['files'][name], f'Saved artifact hash mismatch: {name}')
    require(initial_hashes['training'] == development['training_sha256'] == manifest['training_input']['sha256'], 'Training input hash mismatch.')
    require(paths['training'].stat().st_size == manifest['training_input']['bytes'], 'Training input size mismatch.')
    rule = development['threshold']
    require(rule['value'] == 0.01 and rule['comparison'] == '>=' and rule['selected_on'] == 'validation', 'Fixed validation threshold rule mismatch.')
    require(development['metrics']['selected_threshold'] == rule['value'], 'Recorded metric threshold mismatch.')
    for package in ('xgboost', 'scikit-learn', 'numpy', 'pandas', 'joblib'):
        require(importlib.metadata.version(package) == manifest['provenance']['environment']['packages'][package], f'Runtime version mismatch: {package}')
    features = read_json(paths['feature_names.json'])
    require(isinstance(features, list) and len(features) == expected_features and len(set(features)) == len(features), 'Invalid saved feature schema.')
    X, y = prepare_features(load_raw_aps(paths['training'], features, expected_features))
    with np.load(paths['split_indices.npz'], allow_pickle=False) as split:
        require(set(split.files) == {'train', 'validation'}, 'Unexpected split arrays.')
        train, validation = split['train'], split['validation']
    verify_split(train, validation, y, manifest['split'])
    for name, indices in [('train', train), ('validation', validation)]:
        actual = dict(shape=[len(indices), len(features)], label_counts={str(i): int((y[indices] == i).sum()) for i in (0, 1)})
        require(actual == manifest['data'][name], f'{name} shape/label count mismatch.')
    # Deserialize only this hash-verified, recorded local imputer; never a legacy model pickle.
    imputer = joblib.load(paths['imputer.joblib'])
    require(isinstance(imputer, SimpleImputer) and imputer.strategy == 'median' and not imputer.add_indicator, 'Unsupported saved imputer.')
    verify_feature_order(imputer.feature_names_in_, features)
    require(imputer.n_features_in_ == len(features) and np.isfinite(imputer.statistics_).all(), 'Invalid saved imputer statistics.')
    X_validation = X.iloc[validation]
    matrix = imputer.transform(X_validation)
    require(matrix.shape == X_validation.shape and np.isfinite(matrix).all(), 'Invalid imputed validation matrix.')
    expected = np.where(X_validation.isna(), imputer.statistics_, X_validation.to_numpy())
    require(np.array_equal(matrix, expected), 'Imputer transform differs from stored-median replacement.')
    model = xgb.XGBClassifier()
    model.load_model(paths['model.ubj'])
    booster = model.get_booster()
    configuration = json.loads(booster.save_config())
    require(configuration['learner']['objective']['name'] == 'binary:logistic', 'Expected binary logistic objective.')
    require(booster.num_features() == len(features) and booster.num_boosted_rounds() == manifest['model_params']['n_estimators'], 'Saved model shape/tree count mismatch.')
    if booster.feature_names is not None:
        verify_feature_order(booster.feature_names, features)
    require(np.array_equal(model.classes_, [0, 1]), 'Unexpected class/probability column convention.')
    probability = model.predict_proba(matrix)[:, 1]
    stored, probability_error = verify_predictions(pd.read_csv(paths['validation_predictions.csv']), validation, y[validation], probability, rule['value'])
    predicted = stored >= rule['value']
    masks = dict(TP=(y[validation] == 1) & predicted, FP=(y[validation] == 0) & predicted,
                 FN=(y[validation] == 1) & ~predicted, TN=(y[validation] == 0) & ~predicted)
    for name, mask in masks.items():
        require(int(mask.sum()) == development['metrics']['selected'][name], f'Stored validation {name} count mismatch.')
    # All provenance, schema, split, label and probability checks above precede contributions.
    contributions, reconstruction = native_contributions(booster, matrix, features, stored)
    groups, constant = summarize(X_validation, matrix, contributions, y[validation], stored, rule['value'])
    require(initial_hashes == {name: sha256(path) for name, path in paths.items()}, 'An approved input changed during the audit.')
    sources = [Path(__file__), ROOT/'scripts/audit_phase1_explanations.py', ROOT/'tests/test_phase1_explanations.py',
               ROOT/'src/data_loading.py', ROOT/'src/preprocessing.py', ROOT/'src/config.py']
    return dict(audit='Phase 1 validation model explanations v1', run_id=RUN_ID,
        scope='Exploratory explanations of this classifier on the validation split; threshold-selection data, audited after the official test result. No causal roots or new independent test performance.',
        threshold=rule, validation=dict(samples=len(validation), positive_labels=int(y[validation].sum()), negative_labels=int((y[validation] == 0).sum())),
        verification=dict(approved_input_sha256=initial_hashes, manifest_anchor_sha256=expected_manifest_sha256,
            imputed_matrix=dict(shape=list(matrix.shape), dtype=str(matrix.dtype), sha256_c_order=hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest()),
            maximum_stored_probability_absolute_error=probability_error, reconstruction=reconstruction,
            probability_atol=PROBABILITY_ATOL, probability_rtol=0, margin_atol=MARGIN_ATOL, margin_rtol=MARGIN_RTOL),
        convention=dict(api='Booster.predict(DMatrix(exact imputed validation matrix), pred_contribs=True, approx_contribs=False, strict_shape=True, iteration_range=(0,0))',
            output_shape=[len(validation), 1, len(features)+1], final_column='bias', units='raw positive-class log-odds margin',
            algorithm='XGBoost native exact TreeSHAP using stored tree path statistics; no new background sample',
            positive_meaning='increases class-1 margin relative to bias', ranking='mean absolute contribution; ties by saved feature index',
            top_k=TOP_K, signed_parts='means of max(contribution,0) and min(contribution,0), both over all group rows'),
        provenance=dict(python=platform.python_version(), packages={p: importlib.metadata.version(p) for p in ('xgboost','numpy','pandas','scipy','scikit-learn','joblib')},
            source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in sources}),
        groups=groups, cd_000=constant)
