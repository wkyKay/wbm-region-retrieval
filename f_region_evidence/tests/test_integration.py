from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from f_region_evidence import calibration, pipeline, run, scoring, state
from f_region_evidence.tests.test_scoring import fixture
from wbm_retrieval.retrieval_protocol import FullGalleryPool


def write_cache(folder, cache):
    folder.mkdir(parents=True)
    np.savez(folder/'regions.npz', **{k: v for k, v in cache.items() if k != 'features'})
    np.save(folder/'h512.npy', cache['features'])
    state.seal(folder, 'synthetic-cache')
    return pipeline.read_scoring_cache(folder)


class SpawnIntegrationTests(unittest.TestCase):
    def test_path_only_calibration_and_full_score_workers_replay_direct_provider(self):
        """Synthetic 501-gallery smoke; no formal reports, training or source data."""
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            raw, _ = fixture((1,)*501, start=1000)
            train, _ = fixture((1, 2, 4, 1, 3, 2, 4, 2, 1), start=100)
            cache = write_cache(output/'cache/valid/base4', raw)
            train_cache = write_cache(output/'cache/train', train)
            reference = run.jsonable(calibration.select_reference(train_cache, size=8))
            task = (str(output), 'base4', str(cache['folder']), str(train_cache['folder']),
                    reference, 'valid', 'synthetic-context')
            with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as executor:
                name, thresholds = executor.submit(run._calibrate_one, task).result(timeout=60)
            self.assertEqual(name, 'base4')
            direct, _ = calibration.calibrate(cache, train_cache, reference)
            for key in thresholds:
                np.testing.assert_array_equal(thresholds[key], direct[key])
            ids = raw['map_ids']
            pool = FullGalleryPool(ids[:3], ids, 'valid', 'synthetic-full-gallery')
            specs = {name: spec for name, spec in state.grid().items() if spec['proposal'] == 'base4'}
            # This rejects accidental spawn pickling of the mapped source h512.
            with mock.patch.object(np.memmap, '__reduce__', side_effect=AssertionError('Do not pickle memmaps')):
                providers = pipeline.compute_scores(output, {'base4': cache}, pool, {'base4': thresholds},
                                                    'synthetic-context', specs, workers=2)
            expected = scoring.Provider(cache, pool, thresholds=thresholds)
            for variant, provider in providers.items():
                for query in pool:
                    candidates, scores = expected.score(query, specs[variant]['rejection'])
                    np.testing.assert_array_equal(provider.scores(query, candidates), scores)
                    self.assertEqual(provider.matrix[provider.query_positions[int(query)],
                                                     provider.gallery_positions[int(query)]], 0.)
            with mock.patch.object(pipeline, 'ProcessPoolExecutor', side_effect=AssertionError('Reuse sealed scores')):
                reused = pipeline.compute_scores(output, {'base4': cache}, pool, {'base4': thresholds},
                                                 'synthetic-context', specs, workers=2)
            self.assertEqual(set(reused), set(specs))


if __name__ == '__main__':
    unittest.main()
