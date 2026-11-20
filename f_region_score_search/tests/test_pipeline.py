"""Synthetic full-gallery checks, including spawned report/verification workers."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from f_region_structure.pipeline import Provider as OldProvider, _scalar_greedy
from f_region_structure.tests.test_pipeline import fixture, MATCHING
from wbm_retrieval.common import read_json, file_hash
from f_region_score_search import pipeline, scoring
from f_region_score_search.state import verified


class ProviderTests(unittest.TestCase):
    def test_all_variants_preserve_padding_empty_queries_and_old_product(self):
        cache, pool, _, _ = fixture()
        new = pipeline.Provider(cache, pool, MATCHING)
        old = OldProvider(cache, pool, MATCHING)
        for q in pool:
            ids, scores = new.score(q, scoring.frozen_grid()['product'])
            old_ids, old_scores = old.score(q, scoring.OLD_SPEC)
            np.testing.assert_array_equal(ids, old_ids)
            np.testing.assert_array_equal(scores, old_scores)
            qt = new.tokens(q)
            parts = new.parts(q)
            for spec in scoring.frozen_grid().values():
                ids, scores = new.score(q, spec, parts)
                pairs = scoring.score_components(parts, spec)
                expected = [_scalar_greedy(pairs[:, new.tokens(c)], cache['area'][qt]) for c in ids]
                np.testing.assert_allclose(scores, expected, atol=1e-12, rtol=0)
                for c, value in zip(ids, scores):
                    if len(qt) == 0 or len(new.tokens(c)) == 0:
                        self.assertEqual(value, 0.)

    def test_test_split_is_rejected(self):
        cache, pool, _, _ = fixture('test')
        with self.assertRaisesRegex(ValueError, 'only permits valid'):
            pipeline.Provider(cache, pool, MATCHING)


class FullEvaluationTests(unittest.TestCase):
    def test_serial_and_spawned_verification_match_and_completed_reuse_is_readonly(self):
        cache, pool, bundle, _ = fixture()
        with tempfile.TemporaryDirectory() as tmp, patch.object(pipeline, 'KS', [1, 5, 10]), redirect_stdout(StringIO()):
            folders = [Path(tmp)/'serial', Path(tmp)/'parallel']
            outputs = []
            for workers, folder in zip([1, 2], folders):
                folder.mkdir()
                outputs.append(pipeline.evaluate(folder, cache, pool, bundle, MATCHING, 'synthetic_context', workers))
                self.assertEqual(set(outputs[-1]), set(scoring.frozen_grid()))
                for name in outputs[-1]:
                    p = folder/'valid'/name
                    self.assertTrue(verified(p))
                    check = read_json(p/'verification.json')
                    self.assertEqual(check['num_queries'], 3)
                    self.assertEqual(check['ranking_rows'], 30)
                    self.assertEqual(check['full_candidates_checked'], 303)
                    self.assertEqual(check['max_saved_score_absolute_error'], 0.)
            for name in scoring.frozen_grid():
                self.assertEqual(outputs[0][name]['top10_hits'], outputs[1][name]['top10_hits'])
                self.assertEqual(file_hash(folders[0]/'valid'/name/'rankings.csv'),
                                 file_hash(folders[1]/'valid'/name/'rankings.csv'))
                self.assertEqual(outputs[0][name]['metrics'], outputs[1][name]['metrics'])
            before = {str(p): (file_hash(p), p.stat().st_mtime_ns) for p in folders[0].rglob('*') if p.is_file()}
            with patch.object(pipeline, '_finish', side_effect=AssertionError('completed report was rebuilt')):
                pipeline.evaluate(folders[0], cache, pool, bundle, MATCHING, 'synthetic_context', 1)
            after = {str(p): (file_hash(p), p.stat().st_mtime_ns) for p in folders[0].rglob('*') if p.is_file()}
            self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
