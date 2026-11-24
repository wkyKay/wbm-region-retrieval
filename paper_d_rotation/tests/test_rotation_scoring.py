"""Current-primary arithmetic and source-feature regression checks."""
import unittest

import numpy as np

from paper_c_rotation import scoring as ablation
from paper_c_rotation.tests.test_scoring import fixture
from paper_d_rotation import scoring
from paper_d_rotation.examples import enrich_explanation


class RotationScoringTests(unittest.TestCase):
    def test_current_primary_pair_arithmetic_matches_c_and_differs_from_old(self):
        parts = {'cosine': np.array([[.81, .02]], dtype=np.float64),
                 'structure_similarity': np.array([[.93, .74]]),
                 'area_similarity': np.array([[.70, .99]])}
        actual = scoring.score_components(parts)
        expected = .85 * (.4 * parts['cosine'] + .6 * parts['structure_similarity']) + .15 * parts['area_similarity']
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual, ablation.score_components(parts, ablation.grid()['primary']))
        historical = .75 * (.4 * parts['cosine'] + .6 * parts['structure_similarity']) + .25 * parts['area_similarity']
        self.assertFalse(np.array_equal(actual, historical))

    def test_new_provider_explanations_reproduce_all_fixture_c_primary_scores(self):
        cache, pool = fixture(dtype=np.float64, split='test')
        provider = scoring.Provider(cache, pool)
        reference = ablation.Provider(cache, pool)
        for query in pool:
            ids, values = provider.score(query)
            expected_ids, expected = reference.score(query, ablation.grid()['primary'])
            np.testing.assert_array_equal(ids, expected_ids)
            np.testing.assert_array_equal(values, expected)
            for candidate, score in zip(ids, values):
                actual = provider.explain(query, int(candidate))
                self.assertAlmostEqual(actual['score'], float(score), places=12)
                if len(provider.tokens(query)):
                    enriched = enrich_explanation(actual,
                        dict(ranking_score=float(score), query_id=int(query), candidate_id=int(candidate)),
                        cache['area'][provider.tokens(query)])
                    self.assertLessEqual(enriched['algebraic_decomposition_error'], 1e-12)


if __name__ == '__main__':
    unittest.main()
