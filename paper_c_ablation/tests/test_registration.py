"""Controls cannot become duplicate experiments or bypass frozen evidence checks."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from paper_c_ablation import register, state
from wbm_retrieval.common import file_hash, read_json, write_json


class RegistrationTests(unittest.TestCase):
    def test_eighteen_reports_produce_only_fourteen_new_registered_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root/'results/real_time_run'
            for stage in state.STAGES:
                for name in state.grid():
                    folder = output/stage/name
                    signature = state.evaluation_signature('context', stage, name)
                    ident = state.identity(output, stage, name, signature)
                    state.record(folder/'run_identity.json', ident)
                    state.record(folder/'evaluation_manifest.json', {**ident, 'signature': signature})
                    state.seal(folder, signature)
            with patch.object(register, 'verify_run'):
                result = register.proposal(output, root=root)
            self.assertEqual(len(result['runs']), 14)
            self.assertEqual(len({x['key'] for x in result['runs']}), 14)
            pairs = {(x['protocol'], x['split'], x['variant']) for x in result['runs']}
            self.assertNotIn(('retrieval_v1', 'valid', 'PaperC/no_a'), pairs)
            self.assertIn(('retrieval_v1', 'test', 'PaperC/no_a'), pairs)
            self.assertIn(('coverage_cross_combination_v1', 'test', 'PaperC/no_a'), pairs)
            self.assertTrue(all(x['variant'] != 'PaperC/primary' for x in result['runs']))
            self.assertTrue(all(x['selection'] is None for x in result['runs']))

    def test_formula_audit_cannot_hide_an_incorrect_fusion_formula(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            row = {'query_ids': list(range(37)), 'candidate_pairs': 740,
                   'max_raw_formula_absolute_error': 0., 'max_matching_absolute_error': 0.,
                   'max_fusion_absolute_error': 0.}
            audit = {'status': 'verified', 'candidate_pairs': 4440,
                     'variants': {n: dict(row) for n in state.grid()},
                     'max_raw_formula_absolute_error': 0., 'max_matching_absolute_error': 0.,
                     'max_fusion_absolute_error': 0.}
            path = output/'formula_audit_valid.json'
            write_json(path, audit)
            register._formula(output, 'valid')
            audit['variants']['no_c']['max_fusion_absolute_error'] = 1e-3
            write_json(path, audit)
            with self.assertRaisesRegex(ValueError, 'fusion'):
                register._formula(output, 'valid')

    def test_replay_requires_original_identity_and_exact_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            folder = output/'historical'
            folder.mkdir()
            ranking = folder/'rankings.csv'
            ranking.write_text('historical immutable score rows')
            write_json(folder/'completed.json', {'status': 'completed'})
            source_binding = {str(p): file_hash(p) for p in (ranking, folder/'completed.json')}
            old_key = state.REPLAY_KEYS['test']['primary']
            reference = {'key': old_key, 'folder': str(folder), 'ranking_sha256': file_hash(ranking),
                         'completed_sha256': file_hash(folder/'completed.json')}
            cache = output/'score_cache/test/primary'
            cache.mkdir(parents=True)
            write_json(cache/'completed.json', {'signature': 'cache'})
            write_json(output/'inputs_test.json', {'replay_references': {'test': {'primary': reference}}})
            item = {'status': 'verified', 'source_key': old_key, 'source': str(ranking),
                    'source_sha256': file_hash(ranking), 'source_completed_sha256': file_hash(folder/'completed.json'),
                    'score_cache_sha256': file_hash(cache/'completed.json'), 'num_queries': 7199,
                    'ranking_rows': 7199*500, 'top500_identical': True, 'max_score_error': 0.}
            receipt = {'status': 'verified', 'stage': 'test', 'controls': {'primary': item}}
            write_json(output/'replay_test.json', receipt)
            register._replay(output, 'test', 'context', source_binding)
            item['max_score_error'] = 1e-15
            write_json(output/'replay_test.json', receipt)
            with self.assertRaisesRegex(ValueError, 'historical replay'):
                register._replay(output, 'test', 'context', source_binding)
            item['max_score_error'] = 0.
            item['source_key'] = 'another_source'
            write_json(output/'replay_test.json', receipt)
            with self.assertRaises(ValueError):
                register._replay(output, 'test', 'context', source_binding)


if __name__ == '__main__':
    unittest.main()
