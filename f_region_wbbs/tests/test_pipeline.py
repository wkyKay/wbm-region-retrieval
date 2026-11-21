from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from f_region_wbbs import counts, scoring, pipeline, state
from f_region_wbbs.tests.test_prepare import build
from f_region_wbbs.tests.test_counts import Pool


class PipelineTests(unittest.TestCase):
    def test_real_cache_count_spawn_scores_and_immutable_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache, _, _ = build(tmp, workers=0)
            output = Path(tmp)/'new'
            pool = Pool(cache['map_ids'], queries=cache['map_ids'][[1,0,3]])
            pool.candidates_per_query = len(pool.gallery)-1
            matrices,tokens = counts.create(output,cache,pool,'context',workers=2)
            provider = scoring.Provider(cache,pool,matrices,tokens)
            specs = state.grid()
            saved = pipeline.compute_scores(output,cache,pool,'context',specs,workers=2)
            for q in pool:
                parts=provider.parts(q)
                for name,spec in specs.items():
                    ids,expected = provider.score(q,spec,parts)
                    np.testing.assert_array_equal(saved[name].scores(q,ids),expected)
                    self.assertEqual(saved[name].matrix[saved[name].query_positions[int(q)],saved[name].gallery_positions[int(q)]],0.)
            with patch.object(pipeline,'ProcessPoolExecutor',side_effect=AssertionError('complete cache must reuse')):
                reused=pipeline.compute_scores(output,cache,pool,'context',specs,workers=2)
            self.assertEqual(set(reused),set(specs))
            with self.assertRaises(ValueError):
                pipeline.compute_scores(output,cache,pool,'changed_context',specs,workers=2)


if __name__=='__main__':unittest.main()
