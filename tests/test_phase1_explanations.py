"""Tiny synthetic saved-run fixtures; never read real APS data in these tests."""
import builtins
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import xgboost as xgb

from scripts import audit_phase1_explanations as cli
from src import phase1_explanation_audit as audit


class Phase1ExplanationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.paths = audit.input_paths(self.root)
        for path in self.paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(17)
        self.X = pd.DataFrame(rng.normal(size=(80, 3)), columns=['a', 'b', 'c'])
        self.y = (self.X.a.to_numpy() > 0).astype(int)
        self.X['cd_000'] = 1209600.
        self.X.loc[[2, 9, 40], 'cd_000'] = np.nan
        self.X.loc[[3, 18, 50], 'b'] = np.nan
        pd.concat([pd.Series(np.where(self.y, 'pos', 'neg'), name='class'), self.X], axis=1).to_csv(self.paths['training'], index=False, na_rep='na')
        # Read back the serialized fixture before synthetic-only fitting.
        self.X, self.y = audit.prepare_features(audit.load_raw_aps(self.paths['training'], expected_features=4))
        self.train, self.val = train_test_split(np.arange(80), test_size=0.25, stratify=self.y, random_state=42)
        self.imputer = SimpleImputer(strategy='median').fit(self.X.iloc[self.train])
        self.matrix = self.imputer.transform(self.X.iloc[self.val])
        self.model = xgb.XGBClassifier(n_estimators=8, max_depth=2, n_jobs=1, random_state=3)
        self.model.fit(self.imputer.transform(self.X.iloc[self.train]), self.y[self.train])
        self.model.save_model(self.paths['model.ubj'])
        joblib.dump(self.imputer, self.paths['imputer.joblib'])
        np.savez(self.paths['split_indices.npz'], train=self.train, validation=self.val)
        self.paths['feature_names.json'].write_text(json.dumps(list(self.X.columns)))
        probability = self.model.predict_proba(self.matrix)[:, 1]
        pd.DataFrame(dict(row_index=self.val, true=self.y[self.val], probability=probability,
                          predicted=(probability >= 0.01).astype(int), predicted_0_5=(probability >= 0.5).astype(int))).to_csv(self.paths['validation_predictions.csv'], index=False)
        groups = dict(TP=int(((self.y[self.val] == 1) & (probability >= 0.01)).sum()),
                      FP=int(((self.y[self.val] == 0) & (probability >= 0.01)).sum()),
                      FN=int(((self.y[self.val] == 1) & (probability < 0.01)).sum()),
                      TN=int(((self.y[self.val] == 0) & (probability < 0.01)).sum()))
        self.manifest = dict(run_id=audit.RUN_ID, status='development_complete',
            data={name: dict(shape=[len(idx), 4], label_counts={str(i): int((self.y[idx] == i).sum()) for i in (0,1)}) for name,idx in [('train',self.train),('validation',self.val)]},
            split=dict(seed=42, validation_size=0.25, stratified=True, index_definition='zero-based data row, excluding CSV header'),
            model_params=dict(n_estimators=8), imputation='median fitted on train only',
            training_input=dict(sha256=audit.sha256(self.paths['training']), bytes=self.paths['training'].stat().st_size),
            files={n:audit.sha256(self.paths[n]) for n in audit.ARTIFACT_NAMES},
            provenance=dict(environment=dict(packages={p:audit.importlib.metadata.version(p) for p in ('numpy','pandas','xgboost','joblib','scikit-learn')})))
        self.development = dict(run_id=audit.RUN_ID, stage='development',
            **{k:self.manifest[k] for k in ('data','split','model_params','imputation')},
            training_sha256=self.manifest['training_input']['sha256'],
            threshold=dict(value=0.01, comparison='>=', selected_on='validation'),
            metrics=dict(selected_threshold=0.01, selected=groups))
        self.seal()

    def seal(self):
        self.paths['manifest'].write_text(json.dumps(self.manifest))
        self.anchor = audit.sha256(self.paths['manifest'])
        self.development['manifest_sha256'] = self.anchor
        self.paths['development'].write_text(json.dumps(self.development))

    def run_audit(self):
        return audit.audit(self.root, expected_manifest_sha256=self.anchor, expected_features=4)

    def test_native_additivity_and_probability_with_real_small_xgboost(self):
        prob = self.model.predict_proba(self.matrix)[:,1]
        contributions, check = audit.native_contributions(self.model.get_booster(), self.matrix, list(self.X.columns), prob)
        self.assertEqual(contributions.shape, (20,5))
        self.assertEqual(check['samples'],20)
        np.testing.assert_array_equal(contributions[:,3],0)
        np.testing.assert_allclose(expit(contributions.astype(float).sum(axis=1)),prob,atol=1e-7,rtol=0)
        with self.assertRaisesRegex(ValueError,'reconstruct'):
            audit.verify_additivity(contributions+1, np.zeros(20),prob)
        with self.assertRaisesRegex(ValueError,'probability'):
            audit.verify_additivity(np.zeros((2,3)), np.zeros(2),np.zeros(2))

    def test_full_audit_read_only_allowlist_and_no_fit(self):
        before = {p:audit.sha256(p) for p in self.root.rglob('*') if p.is_file()}
        allowed = {p.resolve() for p in self.paths.values()}
        # Runtime version metadata is provenance, not an additional data input.
        for package in ('numpy','pandas','xgboost','joblib','scikit-learn','scipy'):
            distribution = audit.importlib.metadata.distribution(package)
            allowed.update(Path(distribution.locate_file(f)).resolve() for f in distribution.files
                           if f.name == 'METADATA')
        source_root = audit.ROOT
        allowed.update(p.resolve() for p in [Path(audit.__file__), source_root/'scripts/audit_phase1_explanations.py',
            source_root/'tests/test_phase1_explanations.py', source_root/'src/data_loading.py', source_root/'src/preprocessing.py', source_root/'src/config.py'])
        real_open, real_io = builtins.open, io.open
        def guarded(fn):
            def checked(path, mode='r', *args, **kwargs):
                if isinstance(path, (str, Path)):
                    self.assertFalse(any(c in mode for c in 'wax+'), f'Unexpected write: {path}')
                    self.assertIn(Path(path).resolve(), allowed, f'Unexpected read: {path}')
                return fn(path, mode, *args, **kwargs)
            return checked
        with patch('builtins.open',guarded(real_open)), patch('io.open',guarded(real_io)), \
             patch.object(SimpleImputer,'fit',side_effect=AssertionError('fit forbidden')), \
             patch.object(xgb.XGBClassifier,'fit',side_effect=AssertionError('fit forbidden')):
            result = self.run_audit()
        self.assertEqual(before,{p:audit.sha256(p) for p in self.root.rglob('*') if p.is_file()})
        self.assertEqual(result['validation']['samples'],20)
        self.assertEqual(result['cd_000']['maximum_absolute_contribution'],0)
        # No test CSV or test artifacts exist anywhere in the fixture.
        self.assertEqual(set(result['verification']['approved_input_sha256']),set(self.paths))

    def test_manifest_artifact_and_training_hash_fail_before_deserialization(self):
        for key in ('manifest','model.ubj','imputer.joblib','training','validation_predictions.csv'):
            with self.subTest(key=key):
                original=self.paths[key].read_bytes()
                self.paths[key].write_bytes(original+b' ')
                try:
                    with patch.object(joblib,'load',side_effect=AssertionError('must not load')), \
                         patch.object(audit,'native_contributions',side_effect=AssertionError('must not explain')):
                        with self.assertRaisesRegex(ValueError,'[Hh]ash'):
                            self.run_audit()
                finally:
                    self.paths[key].write_bytes(original)

    def test_schema_failure_even_with_consistent_artifact_hash(self):
        self.paths['feature_names.json'].write_text(json.dumps(list(self.X.columns)[::-1]))
        self.manifest['files']['feature_names.json']=audit.sha256(self.paths['feature_names.json'])
        self.seal()
        with patch.object(joblib,'load',side_effect=AssertionError('must not load')):
            with self.assertRaisesRegex(ValueError,'names/order'):
                self.run_audit()
        with self.assertRaisesRegex(ValueError,'names/order'):
            audit.verify_feature_order(['b','a'],['a','b'])

    def test_split_label_and_probability_mismatch_precedes_explanations(self):
        frame=pd.read_csv(self.paths['validation_predictions.csv'])
        for column, value, message in [('row_index', -1, 'indices'), ('true', 1-frame.true.iloc[0], 'labels'), ('probability', 0.12345, 'probability')]:
            edited=frame.copy()
            edited.loc[0,column]=value
            edited.to_csv(self.paths['validation_predictions.csv'],index=False)
            self.manifest['files']['validation_predictions.csv']=audit.sha256(self.paths['validation_predictions.csv'])
            self.seal()
            with patch.object(audit,'native_contributions',side_effect=AssertionError('must not explain')):
                with self.assertRaisesRegex(ValueError,message):
                    self.run_audit()
        with self.assertRaisesRegex(ValueError,'overlap'):
            audit.verify_split(self.train,np.repeat(self.val[0],len(self.val)),self.y,self.manifest['split'])
        with self.assertRaisesRegex(ValueError,'split order'):
            audit.verify_split(self.train,self.val[::-1],self.y,self.manifest['split'])

    def test_absolute_ranking_signed_parts_missingness_and_small_group_counts(self):
        raw=pd.DataFrame({'a':[1,np.nan,2,3], 'cd_000':[7,7,np.nan,7]})
        contributions=np.array([[3,0,1],[-3,0,1],[1,0,1],[2,0,1]],dtype=float)
        groups,constant=audit.summarize(raw,raw.fillna(7).to_numpy(),contributions,
                                       np.array([1,1,1,0]),np.array([.1,.2,.001,.2]),.01)
        tp=groups['TP']['top_features'][0]
        self.assertEqual(tp['samples'],2)
        self.assertEqual(tp['mean_absolute_contribution'],3)
        self.assertEqual(tp['mean_signed_contribution'],0)
        self.assertEqual(tp['mean_positive_part'],1.5)
        self.assertEqual(tp['mean_negative_part'],-1.5)
        self.assertEqual(tp['raw_missing_fraction'],.5)
        self.assertEqual(groups['FN']['samples'],1)
        self.assertEqual(groups['TN']['samples'],0)
        self.assertEqual(groups['TN']['top_features'],[])
        self.assertEqual(constant['by_group']['FN']['samples'],1)

    def test_cli_compare_never_writes_and_initial_output_is_exclusive(self):
        output=self.root/'audit.json'
        output.write_text('{"result":1}\n')
        before=output.read_bytes()
        with patch.object(cli,'OUTPUT',output),patch.object(cli,'audit',return_value={'result':1}):
            with patch('sys.argv',['audit','--mode','compare']):
                cli.main()
            with patch('sys.argv',['audit','--mode','write']):
                with self.assertRaises(FileExistsError):
                    cli.main()
        self.assertEqual(output.read_bytes(),before)
        with patch.object(cli,'OUTPUT',output),patch.object(cli,'audit',return_value={'result':2}),patch('sys.argv',['audit','--mode','compare']):
            with self.assertRaises(SystemExit):
                cli.main()
        self.assertEqual(output.read_bytes(),before)


if __name__ == '__main__':
    unittest.main()
